#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from util import set_gpu
# set_gpu(True)

import numpy as np
import matplotlib.pyplot as plt
from time import time
from util import progress_bar, compute_quality_metrics, evaluate_trajectory
from util import compute_absolute_trajectory_error
from transform2d import compose_trajectory
from datasimulator import DataSimulator
from graphoptimizer import GraphOptimizer
from imagematcher import ImageMatcher
from swinloopmodel2 import SwinLoopModel
import torch

# Path config
PATH_DATASET = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET"
PATH_SWIN_MODEL = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS_REAL/swin_classifier_best.pth"

# Flags
ENABLE_NN = True
ENABLE_RANSAC = True
ENABLE_REJECTION = True
MOTION_ESTIMATOR = 1
LOOP_MARGIN = 1
LOOP_SEPARATION = 1
PLOT_ONLINE = False

# Image config
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_NCHAN = 3
PIX_TO_WORLD = np.array([10, 10, 1]).reshape((3, 1))
DS_NOISES = [[0.625, 3.14 / (180 * 4)], [2.5, 3.14 / 180], [5, 2 * 3.14 / 180]]
DS_NOISELEVEL = 0

# Timing
timeNN = timeRANSAC = timeFilter = timeOptimizer = 0

# Objects
sim = DataSimulator(PATH_DATASET, loadImages=True, minOverlap=1,
                    motionSigma=DS_NOISES[DS_NOISELEVEL][0],
                    angleSigma=DS_NOISES[DS_NOISELEVEL][1])
odoCovariance = loopCovariance = sim.odoCovariance

loopModel = SwinLoopModel(PATH_SWIN_MODEL)
matcher = ImageMatcher()

preID, preImage = sim.get_image()
allID, allImages = [preID], [preImage]
theLoops = np.empty((6, 0), dtype='int')
optimizer = GraphOptimizer(initialID=preID, minLoops=5, doFilter=ENABLE_REJECTION)
if PLOT_ONLINE:
    plt.figure()

while sim.update():
    curID, curImage = sim.get_image()
    _, theMotion, _ = sim.match_images(preID, curID, addNoise=True, simulateFailures=False)
    optimizer.add_odometry(curID, theMotion.reshape((3, 1)), odoCovariance)

    candidateIDs = allID[-50:-LOOP_MARGIN:LOOP_SEPARATION]
    if len(candidateIDs) > 0:
        candidateImages = np.array(allImages[:-LOOP_MARGIN:LOOP_SEPARATION])

        if ENABLE_NN:
            curRep = np.repeat(curImage.reshape((1, IMG_HEIGHT, IMG_WIDTH, IMG_NCHAN)),
                               candidateImages.shape[0], axis=0)

            currentPredictions = np.zeros(candidateImages.shape[0], dtype=int)
            BATCH_SIZE = 10
            tStart = time()
            for start in range(0, len(candidateIDs), BATCH_SIZE):
                end = start + BATCH_SIZE
                curRepBatch = curRep[start:end]
                candBatch = candidateImages[start:end]
                preds = loopModel.predict(curRepBatch, candBatch)
                currentPredictions[start:end] = preds
                torch.cuda.empty_cache()
            tEnd = time()
            timeNN += (tEnd - tStart)

        for i, candID in enumerate(candidateIDs):
            curPrediction, motion, gtMatch = sim.match_images(candID, curID, addNoise=True, simulateFailures=True)
            if ENABLE_NN:
                curPrediction = currentPredictions[i]

            loopStats = [candID, curID, gtMatch, curPrediction, curPrediction, 0]

            if curPrediction:
                matcher.define_images(candidateImages[i], curImage)
                if ENABLE_RANSAC or MOTION_ESTIMATOR == 1:
                    tStart = time()
                    matcher.estimate()
                    tEnd = time()
                if ENABLE_RANSAC:
                    loopStats[4] = int(not matcher.hasFailed)
                    timeRANSAC += (tEnd - tStart)
                if MOTION_ESTIMATOR == 1:
                    motion = matcher.theMotion
                if not ENABLE_RANSAC or not matcher.hasFailed:
                    motion = motion.reshape((3, 1)) * PIX_TO_WORLD
                    tStart = time()
                    _, _ = optimizer.add_loop(candID, curID, motion, loopCovariance)
                    tEnd = time()
                    if ENABLE_REJECTION:
                        timeFilter += (tEnd - tStart)

            theLoops = np.concatenate((theLoops, np.array(loopStats).reshape(6, 1)), axis=1)

        tStart = time()
        selected = optimizer.validate()
        tEnd = time()
        if ENABLE_REJECTION:
            timeFilter += (tEnd - tStart)

        tStart = time()
        optimizer.optimize()
        tEnd = time()
        timeOptimizer += (tEnd - tStart)

        for sel in selected:
            for i in range(theLoops.shape[1]):
                if theLoops[0, i] == sel[0] and theLoops[1, i] == sel[1]:
                    theLoops[5, i] = 1
                    break

    allID.append(curID)
    allImages.append(curImage)
    preID, preImage = curID, curImage

    if PLOT_ONLINE:
        plt.cla()
        optimizer.plot(plotLoops=True, mainStep=10, secondaryStep=0)
        plt.axis('equal')
        plt.show()
        plt.pause(.05)

    progress_bar(sim.curStep - sim.startStep, sim.endStep - sim.startStep)

print('\nQuick loop summary:')
print('Detected loops (NN only):', np.sum(theLoops[3, :]))
print('Accepted loops (final):', np.sum(theLoops[5, :]))

plt.figure()
plt.cla()
optimizer.plot(plotLoops=True, mainStep=1, secondaryStep=20)
plt.axis('equal')
plt.show()

nnTP = np.sum((theLoops[2, :] == 1) & (theLoops[3, :] == 1))
nnFP = np.sum((theLoops[2, :] == 0) & (theLoops[3, :] == 1))
nnTN = np.sum((theLoops[2, :] == 0) & (theLoops[3, :] == 0))
nnFN = np.sum((theLoops[2, :] == 1) & (theLoops[3, :] == 0))
nnAcc, nnPrec, nnRec, nnFall = compute_quality_metrics(nnTP, nnFP, nnTN, nnFN)

rTP = np.sum((theLoops[2, :] == 1) & (theLoops[4, :] == 1))
rFP = np.sum((theLoops[2, :] == 0) & (theLoops[4, :] == 1))
rTN = np.sum((theLoops[2, :] == 0) & (theLoops[4, :] == 0))
rFN = np.sum((theLoops[2, :] == 1) & (theLoops[4, :] == 0))
rAcc, rPrec, rRec, rFall = compute_quality_metrics(rTP, rFP, rTN, rFN)

fTP = np.sum((theLoops[2, :] == 1) & (theLoops[5, :] == 1))
fFP = np.sum((theLoops[2, :] == 0) & (theLoops[5, :] == 1))
fTN = np.sum((theLoops[2, :] == 0) & (theLoops[5, :] == 0))
fFN = np.sum((theLoops[2, :] == 1) & (theLoops[5, :] == 0))
fAcc, fPrec, fRec, fFall = compute_quality_metrics(fTP, fFP, fTN, fFN)

print('\n                NN          RANSAC      FILTER')
print('TP        : %8d -> %8d -> %8d' % (nnTP, rTP, fTP))
print('FP        : %8d -> %8d -> %8d' % (nnFP, rFP, fFP))
print('TN        : %8d -> %8d -> %8d' % (nnTN, rTN, fTN))
print('FN        : %8d -> %8d -> %8d' % (nnFN, rFN, fFN))
print('ACCURACY  : %8.3f -> %8.3f -> %8.3f' % (nnAcc, rAcc, fAcc))
print('PRECISION : %8.3f -> %8.3f -> %8.3f' % (nnPrec, rPrec, fPrec))
print('RECALL    : %8.3f -> %8.3f -> %8.3f' % (nnRec, rRec, fRec))
print('FALLOUT   : %8.3f -> %8.3f -> %8.3f' % (nnFall, rFall, fFall))

print('\n                TIME CONSUMPTION')
print('NN        : %8d s' % timeNN)
print('RANSAC    : %8d s' % timeRANSAC)
print('Filtering : %8d s' % timeFilter)
print('Optimizer : %8d s' % timeOptimizer)

print('\n[EVALUATING TRAJECTORY]')
traj = np.array([np.array(v.pose) for v in optimizer.theVertices]).T
gtOdom = sim.gtOdom[0:3, 1:]
avgError = evaluate_trajectory(traj, gtOdom, True)
print('\n * AVERAGE ERROR : %5.3f UNITS PER TRAVELLED UNIT' % avgError)
print('[EVALUATION DONE]')

print('\n[COMPUTING ABSOLUTE TRAJECTORY ERROR]')
errors, meanErr, stdErr = compute_absolute_trajectory_error(traj, compose_trajectory(gtOdom))
plt.figure()
plt.plot(errors)
plt.xlabel('Time step')
plt.ylabel('Absolute error')
plt.show()
print(' * MEAN OF TRAJECTORY ABSOLUTE ERROR %5.3f' % meanErr)
print(' * STANDARD DEVIATION OF TRAJECTORY ABSOLUTE ERROR %5.3f' % stdErr)
print('[ATE COMPUTED]')



# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# #from util import set_gpu
# #set_gpu(True)

# import numpy as np
# import matplotlib.pyplot as plt
# from time import time
# from util import progress_bar, compute_quality_metrics, evaluate_trajectory
# from util import compute_absolute_trajectory_error
# from transform2d import compose_trajectory
# from datasimulator import DataSimulator
# from graphoptimizer import GraphOptimizer
# from imagematcher import ImageMatcher
# from swinloopmodel2 import SwinLoopModel  # You must create this file
# import torch

# # Path config
# PATH_DATASET = "/home/svillhauer/Desktop/AI_project/UCAMGEN-main/REALDATASET"
# PATH_SWIN_MODEL = "/home/svillhauer/Desktop/AI_project/SNNLOOP-main/TRAN_RESULTS_REAL/swin_classifier_best.pth"

# # Flags
# ENABLE_NN = True
# ENABLE_RANSAC = True
# ENABLE_REJECTION = True
# MOTION_ESTIMATOR = 1
# LOOP_MARGIN = 1
# LOOP_SEPARATION = 1
# PLOT_ONLINE = False

# # Image config
# IMG_WIDTH = 64
# IMG_HEIGHT = 64
# IMG_NCHAN = 3
# PIX_TO_WORLD = np.array([10, 10, 1]).reshape((3, 1))
# DS_NOISES = [[0.625, 3.14 / (180 * 4)], [2.5, 3.14 / 180], [5, 2 * 3.14 / 180]]
# DS_NOISELEVEL = 0

# # Timing
# timeNN = timeRANSAC = timeFilter = timeOptimizer = 0

# # Objects
# sim = DataSimulator(PATH_DATASET, loadImages=True, minOverlap=1,
#                     motionSigma=DS_NOISES[DS_NOISELEVEL][0],
#                     angleSigma=DS_NOISES[DS_NOISELEVEL][1])
# odoCovariance = loopCovariance = sim.odoCovariance

# loopModel = SwinLoopModel(PATH_SWIN_MODEL)
# matcher = ImageMatcher()

# preID, preImage = sim.get_image()
# allID, allImages = [preID], [preImage]
# theLoops = np.empty((6, 0), dtype='int')
# optimizer = GraphOptimizer(initialID=preID, minLoops=5, doFilter=ENABLE_REJECTION)
# if PLOT_ONLINE:
#     plt.figure()

# while sim.update():
#     curID, curImage = sim.get_image()
#     _, theMotion, _ = sim.match_images(preID, curID, addNoise=True, simulateFailures=False)
#     optimizer.add_odometry(curID, theMotion.reshape((3, 1)), odoCovariance)

#     candidateIDs = allID[-50:-LOOP_MARGIN:LOOP_SEPARATION]
#     if len(candidateIDs) > 0:
#         candidateImages = np.array(allImages[:-LOOP_MARGIN:LOOP_SEPARATION])

#         if ENABLE_NN:
#             curRep = np.repeat(curImage.reshape((1, IMG_HEIGHT, IMG_WIDTH, IMG_NCHAN)),
#                                candidateImages.shape[0], axis=0)
#             tStart = time()
#             currentPredictions = loopModel.predict(curRep, candidateImages)
#             tEnd = time()
#             timeNN += (tEnd - tStart)

#         for i, candID in enumerate(candidateIDs):
#             curPrediction, motion, gtMatch = sim.match_images(candID, curID, addNoise=True, simulateFailures=True)
#             if ENABLE_NN:
#                 curPrediction = currentPredictions[i]

#             loopStats = [candID, curID, gtMatch, curPrediction, curPrediction, 0]

#             if curPrediction:
#                 matcher.define_images(candidateImages[i], curImage)
#                 if ENABLE_RANSAC or MOTION_ESTIMATOR == 1:
#                     tStart = time()
#                     matcher.estimate()
#                     tEnd = time()
#                 if ENABLE_RANSAC:
#                     loopStats[4] = int(not matcher.hasFailed)
#                     timeRANSAC += (tEnd - tStart)
#                 if MOTION_ESTIMATOR == 1:
#                     motion = matcher.theMotion
#                 if not ENABLE_RANSAC or not matcher.hasFailed:
#                     motion = motion.reshape((3, 1)) * PIX_TO_WORLD
#                     tStart = time()
#                     _, _ = optimizer.add_loop(candID, curID, motion, loopCovariance)
#                     tEnd = time()
#                     if ENABLE_REJECTION:
#                         timeFilter += (tEnd - tStart)

#             theLoops = np.concatenate((theLoops, np.array(loopStats).reshape(6, 1)), axis=1)

#         tStart = time()
#         selected = optimizer.validate()
#         tEnd = time()
#         if ENABLE_REJECTION:
#             timeFilter += (tEnd - tStart)

#         tStart = time()
#         optimizer.optimize()
#         tEnd = time()
#         timeOptimizer += (tEnd - tStart)

#         for sel in selected:
#             for i in range(theLoops.shape[1]):
#                 if theLoops[0, i] == sel[0] and theLoops[1, i] == sel[1]:
#                     theLoops[5, i] = 1
#                     break

#     allID.append(curID)
#     allImages.append(curImage)
#     preID, preImage = curID, curImage

#     if PLOT_ONLINE:
#         plt.cla()
#         optimizer.plot(plotLoops=True, mainStep=10, secondaryStep=0)
#         plt.axis('equal')
#         plt.show()
#         plt.pause(.05)

#     progress_bar(sim.curStep - sim.startStep, sim.endStep - sim.startStep)

# print('\nQuick loop summary:')
# print('Detected loops (NN only):', np.sum(theLoops[3, :]))
# print('Accepted loops (final):', np.sum(theLoops[5, :]))

# plt.figure()
# plt.cla()
# optimizer.plot(plotLoops=True, mainStep=1, secondaryStep=20)
# plt.axis('equal')
# plt.show()

# nnTP = np.sum((theLoops[2, :] == 1) & (theLoops[3, :] == 1))
# nnFP = np.sum((theLoops[2, :] == 0) & (theLoops[3, :] == 1))
# nnTN = np.sum((theLoops[2, :] == 0) & (theLoops[3, :] == 0))
# nnFN = np.sum((theLoops[2, :] == 1) & (theLoops[3, :] == 0))
# nnAcc, nnPrec, nnRec, nnFall = compute_quality_metrics(nnTP, nnFP, nnTN, nnFN)

# rTP = np.sum((theLoops[2, :] == 1) & (theLoops[4, :] == 1))
# rFP = np.sum((theLoops[2, :] == 0) & (theLoops[4, :] == 1))
# rTN = np.sum((theLoops[2, :] == 0) & (theLoops[4, :] == 0))
# rFN = np.sum((theLoops[2, :] == 1) & (theLoops[4, :] == 0))
# rAcc, rPrec, rRec, rFall = compute_quality_metrics(rTP, rFP, rTN, rFN)

# fTP = np.sum((theLoops[2, :] == 1) & (theLoops[5, :] == 1))
# fFP = np.sum((theLoops[2, :] == 0) & (theLoops[5, :] == 1))
# fTN = np.sum((theLoops[2, :] == 0) & (theLoops[5, :] == 0))
# fFN = np.sum((theLoops[2, :] == 1) & (theLoops[5, :] == 0))
# fAcc, fPrec, fRec, fFall = compute_quality_metrics(fTP, fFP, fTN, fFN)

# print('\n                NN          RANSAC      FILTER')
# print('TP        : %8d -> %8d -> %8d' % (nnTP, rTP, fTP))
# print('FP        : %8d -> %8d -> %8d' % (nnFP, rFP, fFP))
# print('TN        : %8d -> %8d -> %8d' % (nnTN, rTN, fTN))
# print('FN        : %8d -> %8d -> %8d' % (nnFN, rFN, fFN))
# print('ACCURACY  : %8.3f -> %8.3f -> %8.3f' % (nnAcc, rAcc, fAcc))
# print('PRECISION : %8.3f -> %8.3f -> %8.3f' % (nnPrec, rPrec, fPrec))
# print('RECALL    : %8.3f -> %8.3f -> %8.3f' % (nnRec, rRec, fRec))
# print('FALLOUT   : %8.3f -> %8.3f -> %8.3f' % (nnFall, rFall, fFall))

# print('\n                TIME CONSUMPTION')
# print('NN        : %8d s' % timeNN)
# print('RANSAC    : %8d s' % timeRANSAC)
# print('Filtering : %8d s' % timeFilter)
# print('Optimizer : %8d s' % timeOptimizer)

# print('\n[EVALUATING TRAJECTORY]')
# traj = np.array([np.array(v.pose) for v in optimizer.theVertices]).T
# gtOdom = sim.gtOdom[0:3, 1:]
# avgError = evaluate_trajectory(traj, gtOdom, True)
# print('\n * AVERAGE ERROR : %5.3f UNITS PER TRAVELLED UNIT' % avgError)
# print('[EVALUATION DONE]')

# print('\n[COMPUTING ABSOLUTE TRAJECTORY ERROR]')
# errors, meanErr, stdErr = compute_absolute_trajectory_error(traj, compose_trajectory(gtOdom))
# plt.figure()
# plt.plot(errors)
# plt.xlabel('Time step')
# plt.ylabel('Absolute error')
# plt.show()
# print(' * MEAN OF TRAJECTORY ABSOLUTE ERROR %5.3f' % meanErr)
# print(' * STANDARD DEVIATION OF TRAJECTORY ABSOLUTE ERROR %5.3f' % stdErr)
# print('[ATE COMPUTED]')
