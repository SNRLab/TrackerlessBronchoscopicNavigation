import os
from pyexpat import model
import unittest
# from matplotlib.pyplot import get
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from vtk.util import numpy_support
import numpy as np
import torch
import math
from sys import platform
from PIL import Image
from Resources import layers
import re
import json
import time
import cv2
from scipy.spatial.transform import Rotation as R

class ReplayTrackerlessData(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Bakse/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Replay Trackerless Data"
    self.parent.categories = ["Navigation"]
    self.parent.dependencies = []
    self.parent.contributors = ["Franklin King, Megha Kalia"]
    self.parent.helpText = """
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
"""
    # Set module icon from Resources/Icons/<ModuleName>.png
    moduleDir = os.path.dirname(self.parent.path)
    for iconExtension in ['.svg', '.png']:
      iconPath = os.path.join(moduleDir, 'Resources/Icons', self.__class__.__name__ + iconExtension)
      if os.path.isfile(iconPath):
        parent.icon = qt.QIcon(iconPath)
        break

class ReplayTrackerlessDataWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    self.stepCount = 1
    self.stepTimer = qt.QTimer()
    self.stepTimer.timeout.connect(self.onStepImage)
    self.centerlineScaleFactor = 1.0
    self.centerlineStartRadius = 1.0
    self.numberOfFrames = 0
    self.finished = False

  def cleanup(self):
    self.stepTimer.stop()
    self.centerlineScaleFactor = 1.0
    self.centerlineStartRadius = 1.0
    self.numberOfFrames = 0

  def onReload(self,moduleName="ReplayTrackerlessData"):
    self.stepTimer.stop()
    self.centerlineScaleFactor = 1.0
    self.centerlineStartRadius = 1.0
    self.numberOfFrames = 0
    globals()[moduleName] = slicer.util.reloadScriptedModule(moduleName)    

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # IO collapsible button
    self.IOCollapsibleButton = ctk.ctkCollapsibleButton()
    self.IOCollapsibleButton.text = "I/O"
    self.IOCollapsibleButton.collapsed = False
    self.layout.addWidget(self.IOCollapsibleButton)
    IOLayout = qt.QFormLayout(self.IOCollapsibleButton)

    self.imageSelector = slicer.qMRMLNodeComboBox()
    self.imageSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    self.imageSelector.selectNodeUponCreation = False
    self.imageSelector.noneEnabled = True
    self.imageSelector.setMRMLScene(slicer.mrmlScene)
    IOLayout.addRow('Image: ', self.imageSelector)

    self.inputTransformSelector = slicer.qMRMLNodeComboBox()
    self.inputTransformSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.inputTransformSelector.selectNodeUponCreation = False
    self.inputTransformSelector.noneEnabled = False
    self.inputTransformSelector.addEnabled = True
    self.inputTransformSelector.removeEnabled = True
    self.inputTransformSelector.setMRMLScene(slicer.mrmlScene)
    IOLayout.addRow("Camera Transform: ", self.inputTransformSelector)

    # Step mode Recorded Data collapsible button
    self.stepModeCollapsibleButton = ctk.ctkCollapsibleButton()
    self.stepModeCollapsibleButton.text = "Step Mode - Recorded Data"
    self.stepModeCollapsibleButton.collapsed = False
    self.layout.addWidget(self.stepModeCollapsibleButton)

    stepModeLayout = qt.QFormLayout(self.stepModeCollapsibleButton)
    self.pathBox = qt.QLineEdit("D:/Partners HealthCare Dropbox/Franklin King/SNRLabDisk/Projects/CanonProj/TrackerlessNavigation/ExperimentResults/Model_Results/BoxPhantom1/D_SelfSupervised-ArtifactRemoval-PoseLoss-LongtermLoss/Output/1/forward/frequency_3")
    self.pathBox.setReadOnly(False)
    self.pathButton = qt.QPushButton("...")
    self.pathButton.clicked.connect(self.select_directory)
    pathBoxLayout = qt.QHBoxLayout()
    pathBoxLayout.addWidget(self.pathBox)
    pathBoxLayout.addWidget(self.pathButton)
    stepModeLayout.addRow("Data: ", pathBoxLayout)

    self.gtPathBox = qt.QLineEdit("D:/Partners HealthCare Dropbox/Franklin King/SNRLabDisk/Projects/CanonProj/TrackerlessNavigation/ExperimentResults/Validation/BoxPhantom1/Images/1/forward")
    self.gtPathBox.setReadOnly(False)
    self.gtPathButton = qt.QPushButton("...")
    self.gtPathButton.clicked.connect(self.select_directory_gt)
    gtPathBoxLayout = qt.QHBoxLayout()
    gtPathBoxLayout.addWidget(self.gtPathBox)
    gtPathBoxLayout.addWidget(self.gtPathButton)
    stepModeLayout.addRow("Ground Truth: ", gtPathBoxLayout)

    self.methodComboBox = qt.QComboBox()
    # self.methodComboBox.addItem('ICP Only')
    self.methodComboBox.addItem('Recorded AI Pose')
    self.methodComboBox.addItem('Recorded AI Pose with Nudge')
    self.methodComboBox.addItem('Recorded AI Pose with ICP')
    self.methodComboBox.addItem('cGAN with ICP')
    self.methodComboBox.addItem('Ground Truth')
    self.methodComboBox.addItem('None')
    self.methodComboBox.setCurrentIndex(0)
    stepModeLayout.addRow('Registration Method:', self.methodComboBox)

    self.autoInputsButton = qt.QPushButton("Auto Set Input")
    stepModeLayout.addRow(self.autoInputsButton)
    self.autoInputsButton.connect('clicked()', self.onAutoInputs)

    self.loadPredButton = qt.QPushButton("Load Predictions")
    stepModeLayout.addRow(self.loadPredButton)
    self.loadPredButton.connect('clicked()', self.onLoadPred)

    self.stepButton = qt.QPushButton("Step Image")
    stepModeLayout.addRow(self.stepButton)
    self.stepButton.connect('clicked()', self.onStepImage)

    self.resetStepButton = qt.QPushButton("Reset Step Count")
    stepModeLayout.addRow(self.resetStepButton)
    self.resetStepButton.connect('clicked()', self.onResetStepCount)

    self.stepSkipBox = qt.QSpinBox()
    self.stepSkipBox.setSingleStep(1)
    self.stepSkipBox.setMaximum(100)
    self.stepSkipBox.setMinimum(1)
    self.stepSkipBox.value = 3
    stepModeLayout.addRow("Step skip: ", self.stepSkipBox)

    self.stepLabel = qt.QLabel("1")
    stepModeLayout.addRow("Image: ", self.stepLabel)

    self.centerlineScaleLabel = qt.QLabel("1.0")
    stepModeLayout.addRow("Centerline Scale: ", self.centerlineScaleLabel)

    self.stepTimerButton = qt.QPushButton("Step Timer")
    self.stepTimerButton.setCheckable(True)
    stepModeLayout.addRow(self.stepTimerButton)
    self.stepTimerButton.connect('clicked()', self.onStepTimer)

    self.stepFPSBox = qt.QSpinBox()
    self.stepFPSBox.setSingleStep(1)
    self.stepFPSBox.setMaximum(144)
    self.stepFPSBox.setMinimum(1)
    self.stepFPSBox.setSuffix(" FPS")
    self.stepFPSBox.value = 30
    stepModeLayout.addRow(self.stepFPSBox)

    self.scaleSliderWidget = ctk.ctkSliderWidget()
    self.scaleSliderWidget.setDecimals(2)
    self.scaleSliderWidget.minimum = 0.00
    self.scaleSliderWidget.maximum = 1000.00
    self.scaleSliderWidget.singleStep = 0.01
    self.scaleSliderWidget.value = 27.00
    stepModeLayout.addRow("Scale Factor:", self.scaleSliderWidget)

    self.flipPredictionCheckbox = qt.QCheckBox("Flip Prediction")
    self.flipPredictionCheckbox.setChecked(False)
    stepModeLayout.addRow(self.flipPredictionCheckbox)

    self.stepStartOffsetBox = qt.QSpinBox()
    self.stepStartOffsetBox.setSingleStep(1)
    self.stepStartOffsetBox.setMaximum(10000)
    self.stepStartOffsetBox.setMinimum(1)
    self.stepStartOffsetBox.value = 1
    stepModeLayout.addRow("Step Start:", self.stepStartOffsetBox)

    self.rotationScaleSliderWidget = ctk.ctkSliderWidget()
    self.rotationScaleSliderWidget.setDecimals(2)
    self.rotationScaleSliderWidget.minimum = 0.00
    self.rotationScaleSliderWidget.maximum = 100.00
    self.rotationScaleSliderWidget.singleStep = 0.01
    self.rotationScaleSliderWidget.value = 1.00
    stepModeLayout.addRow("Rotation Scale Factor:", self.rotationScaleSliderWidget)

    self.nudgeInterval = qt.QSpinBox()
    self.nudgeInterval.setSingleStep(1)
    self.nudgeInterval.setMaximum(100)
    self.nudgeInterval.setMinimum(1)
    self.nudgeInterval.value = 3
    stepModeLayout.addRow("Nudge Interval: ", self.nudgeInterval)

    self.nudgeFactorWidget = ctk.ctkSliderWidget()
    self.nudgeFactorWidget.setDecimals(2)
    self.nudgeFactorWidget.minimum = 0.00
    self.nudgeFactorWidget.maximum = 100.00
    self.nudgeFactorWidget.singleStep = 0.01
    self.nudgeFactorWidget.value = 0.5
    stepModeLayout.addRow("Nudge Factor:", self.nudgeFactorWidget)

    self.nudgeRotationFactorWidget = ctk.ctkSliderWidget()
    self.nudgeRotationFactorWidget.setDecimals(2)
    self.nudgeRotationFactorWidget.minimum = 0.00
    self.nudgeRotationFactorWidget.maximum = 180.00
    self.nudgeRotationFactorWidget.singleStep = 0.01
    self.nudgeRotationFactorWidget.value = 0.5
    stepModeLayout.addRow("Nudge Rotation Factor (Degrees):", self.nudgeRotationFactorWidget)

    self.nudgeLabel = qt.QLabel("")
    stepModeLayout.addRow(self.nudgeLabel)

    self.centerlineNudgeThresholdWidget = ctk.ctkSliderWidget()
    self.centerlineNudgeThresholdWidget.setDecimals(2)
    self.centerlineNudgeThresholdWidget.minimum = 0.00
    self.centerlineNudgeThresholdWidget.maximum = 180.00
    self.centerlineNudgeThresholdWidget.singleStep = 0.01
    self.centerlineNudgeThresholdWidget.value = 10.00
    stepModeLayout.addRow("Center Line Nudge Threshold:", self.centerlineNudgeThresholdWidget)    

    # Depth map point cloud collapsible button
    self.depthMapModelCollapsibleButton = ctk.ctkCollapsibleButton()
    self.depthMapModelCollapsibleButton.text = "Depth Map Model"
    self.depthMapModelCollapsibleButton.collapsed = False
    self.layout.addWidget(self.depthMapModelCollapsibleButton)
    depthMapModeLayout = qt.QFormLayout(self.depthMapModelCollapsibleButton)    

    self.pointCloudSelector = slicer.qMRMLNodeComboBox()
    self.pointCloudSelector.nodeTypes = ( ("vtkMRMLModelNode"), "" )
    self.pointCloudSelector.selectNodeUponCreation = True
    self.pointCloudSelector.addEnabled = True
    self.pointCloudSelector.removeEnabled = True
    self.pointCloudSelector.renameEnabled = True
    self.pointCloudSelector.noneEnabled = True
    self.pointCloudSelector.showHidden = False
    self.pointCloudSelector.showChildNodeTypes = False
    self.pointCloudSelector.setMRMLScene(slicer.mrmlScene)
    depthMapModeLayout.addRow("Point Cloud Model: ", self.pointCloudSelector)

    self.centerlineScalingCheckBox = qt.QCheckBox("Centerline Scaling")
    self.centerlineScalingCheckBox.setChecked(False)
    depthMapModeLayout.addRow(self.centerlineScalingCheckBox)

    self.centerlineSelector = slicer.qMRMLNodeComboBox()
    self.centerlineSelector.nodeTypes = ( ("vtkMRMLModelNode"), "" )
    self.centerlineSelector.selectNodeUponCreation = True
    self.centerlineSelector.addEnabled = True
    self.centerlineSelector.removeEnabled = True
    self.centerlineSelector.renameEnabled = True
    self.centerlineSelector.noneEnabled = True
    self.centerlineSelector.showHidden = False
    self.centerlineSelector.showChildNodeTypes = False
    self.centerlineSelector.setMRMLScene(slicer.mrmlScene)
    depthMapModeLayout.addRow("Center Line Model: ", self.centerlineSelector)

    self.cameraAirwayPositionSelector = slicer.qMRMLNodeComboBox()
    self.cameraAirwayPositionSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.cameraAirwayPositionSelector.selectNodeUponCreation = False
    self.cameraAirwayPositionSelector.noneEnabled = True
    self.cameraAirwayPositionSelector.addEnabled = True
    self.cameraAirwayPositionSelector.removeEnabled = True
    self.cameraAirwayPositionSelector.setMRMLScene(slicer.mrmlScene)
    depthMapModeLayout.addRow("Camera Airway Position Transform: ", self.cameraAirwayPositionSelector)

    self.maskSelectionComboBox = qt.QComboBox()
    self.maskSelectionComboBox.addItems(['Mask Image', 'Crop'])
    self.maskSelectionComboBox.setCurrentIndex(1)
    self.maskSelectionComboBox.currentIndexChanged.connect(self.onMaskSelection)
    depthMapModeLayout.addRow('Mask method:', self.maskSelectionComboBox)

    self.cornerCutBox = qt.QSpinBox()
    self.cornerCutBox.setSingleStep(1)
    self.cornerCutBox.setMaximum(100)
    self.cornerCutBox.setMinimum(0)
    self.cornerCutBox.setSuffix(" px")
    self.cornerCutBox.value = 40
    self.cornerCutBox.setEnabled(True)
    depthMapModeLayout.addRow("Corner size to cut:", self.cornerCutBox)

    self.borderCutBox = qt.QSpinBox()
    self.borderCutBox.setSingleStep(1)
    self.borderCutBox.setMaximum(100)
    self.borderCutBox.setMinimum(0)
    self.borderCutBox.setSuffix(" px")
    self.borderCutBox.value = 40
    self.borderCutBox.setEnabled(True)
    depthMapModeLayout.addRow("Border size to cut:", self.borderCutBox)    

    self.focalLengthBox = ctk.ctkDoubleSpinBox()
    self.focalLengthBox.maximum = 100000.0
    self.focalLengthBox.minimum = 0.0
    self.focalLengthBox.setDecimals(6)
    self.focalLengthBox.setValue(128.7)
    depthMapModeLayout.addRow("Focal Length:", self.focalLengthBox)

    self.imageScaleBox = ctk.ctkDoubleSpinBox()
    self.imageScaleBox.maximum = 100000.0
    self.imageScaleBox.minimum = 0.0
    self.imageScaleBox.setValue(17.00)
    depthMapModeLayout.addRow("Base Image Scale: ", self.imageScaleBox)

    self.depthScaleBox = ctk.ctkDoubleSpinBox()
    self.depthScaleBox.maximum = 100000.0
    self.depthScaleBox.minimum = 0.0
    self.depthScaleBox.setValue(17.00)
    depthMapModeLayout.addRow("Base Depth Scale: ", self.depthScaleBox)

    self.thresholdBox = qt.QSpinBox()
    self.thresholdBox.setSingleStep(1)
    self.thresholdBox.setMaximum(100)
    self.thresholdBox.setMinimum(0)
    self.thresholdBox.setSuffix("%")
    self.thresholdBox.value = 90
    depthMapModeLayout.addRow("Threshold:", self.thresholdBox)

    # ICP collapsible button
    self.icpCollapsibleButton = ctk.ctkCollapsibleButton()
    self.icpCollapsibleButton.text = "ICP Parameters"
    self.icpCollapsibleButton.collapsed = False
    self.layout.addWidget(self.icpCollapsibleButton)
    icpLayout = qt.QFormLayout(self.icpCollapsibleButton)

    self.baseModelSelector = slicer.qMRMLNodeComboBox()
    self.baseModelSelector.nodeTypes = ( ("vtkMRMLModelNode"), "" )
    self.baseModelSelector.selectNodeUponCreation = False
    self.baseModelSelector.addEnabled = False
    self.baseModelSelector.removeEnabled = True
    self.baseModelSelector.renameEnabled = True
    self.baseModelSelector.noneEnabled = False
    self.baseModelSelector.showHidden = False
    self.baseModelSelector.showChildNodeTypes = False
    self.baseModelSelector.setMRMLScene(slicer.mrmlScene)
    icpLayout.addRow("Base Model: ", self.baseModelSelector)

    self.icpMethodComboBox = qt.QComboBox()
    self.icpMethodComboBox.addItem('Similarity')
    self.icpMethodComboBox.addItem('Rigid')
    self.icpMethodComboBox.setCurrentIndex(1)
    icpLayout.addRow('ICP Method:', self.icpMethodComboBox)

    self.checkMeanDistanceCheckBox = qt.QCheckBox("Check Mean Distance")
    self.checkMeanDistanceCheckBox.setChecked(True)
    icpLayout.addRow(self.checkMeanDistanceCheckBox)

    self.icpIterationsSlider = ctk.ctkSliderWidget()
    self.icpIterationsSlider.decimals = 0
    self.icpIterationsSlider.singleStep = 1
    self.icpIterationsSlider.minimum = 1
    self.icpIterationsSlider.maximum = 10000
    self.icpIterationsSlider.value = 1000
    icpLayout.addRow("Number of Iterations:", self.icpIterationsSlider)

    self.icpLandmarksSlider = ctk.ctkSliderWidget()
    self.icpLandmarksSlider.decimals = 0
    self.icpLandmarksSlider.singleStep = 1
    self.icpLandmarksSlider.minimum = 1
    self.icpLandmarksSlider.maximum = 2000
    self.icpLandmarksSlider.value = 300
    icpLayout.addRow("Number of Landmarks:", self.icpLandmarksSlider)

    self.icpMaxDistanceSlider = ctk.ctkSliderWidget()
    self.icpMaxDistanceSlider.decimals = 5
    self.icpMaxDistanceSlider.singleStep = 0.00001
    self.icpMaxDistanceSlider.minimum = 0.00001
    self.icpMaxDistanceSlider.maximum = 1.0
    self.icpMaxDistanceSlider.value = 0.01
    icpLayout.addRow("Maximum of Distance:", self.icpMaxDistanceSlider)

    self.icpcGANPushSlider = ctk.ctkSliderWidget()
    self.icpcGANPushSlider.decimals = 2
    self.icpcGANPushSlider.singleStep = 0.1
    self.icpcGANPushSlider.minimum = 0.0
    self.icpcGANPushSlider.maximum = 1000.0
    self.icpcGANPushSlider.value = 5.0
    icpLayout.addRow("ICP cGAN Push:", self.icpcGANPushSlider)    

    self.initialICPButton = qt.QPushButton("Initialization ICP")
    icpLayout.addRow(self.initialICPButton)
    self.initialICPButton.connect('clicked()', self.onInitializationICP)

    self.initializationICPTransformSelector = slicer.qMRMLNodeComboBox()
    self.initializationICPTransformSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.initializationICPTransformSelector.selectNodeUponCreation = False
    self.initializationICPTransformSelector.noneEnabled = True
    self.initializationICPTransformSelector.addEnabled = True
    self.initializationICPTransformSelector.removeEnabled = True
    self.initializationICPTransformSelector.setMRMLScene(slicer.mrmlScene)
    icpLayout.addRow("Initialization ICP Transform: ", self.initializationICPTransformSelector)

    # Add vertical spacer
    self.layout.addStretch(1)

  def onAutoInputs(self):
    #self.imageSelector.setCurrentNode(slicer.util.getNode('RGB_1'))
    self.inputTransformSelector.setCurrentNode(slicer.util.getNode('Pose'))
    #self.pointCloudSelector.setCurrentNode(slicer.util.getNode('PointCloud'))
    self.centerlineSelector.setCurrentNode(slicer.util.getNode('CenterlineModel'))
    self.cameraAirwayPositionSelector.setCurrentNode(slicer.util.getNode('AirwayPosition'))
    self.baseModelSelector.setCurrentNode(slicer.util.getNode('airway'))

  def select_directory(self):
    directory = qt.QFileDialog.getExistingDirectory(self.parent, "Select Directory")
    if directory:
      self.pathBox.setText(directory)

  def select_directory_gt(self):
    directory = qt.QFileDialog.getExistingDirectory(self.parent, "Select Directory")
    if directory:
      self.gtPathBox.setText(directory)

  def onStepTimer(self):
    if self.stepTimerButton.isChecked():
      self.stepTimer.start(int(1000/int(self.stepFPSBox.value)))
      self.stepFPSBox.enabled = False
    else:
      self.stepTimer.stop()
      self.stepFPSBox.enabled = True

  def onResetStepCount(self):
    self.stepCount = self.stepStartOffsetBox.value
    self.stepLabel.setText(f'{self.stepStartOffsetBox.value}')

    self.finished = False

    layoutManager = slicer.app.layoutManager()
    red = layoutManager.sliceWidget('Red')
    redLogic = red.sliceLogic()
    redLogic.SetSliceOffset(0)

    green = layoutManager.sliceWidget('Green')
    greenLogic = green.sliceLogic()
    greenLogic.SetSliceOffset(0)
    
    self.numberOfFrames = self.count_GT_files(self.gtPathBox.text)

    self.nudgeLabel.text = ""
    scale = self.scaleSliderWidget.value

    self.previousClosestPoint = [np.nan, np.nan, np.nan, 1]
    centerlinePointNode = slicer.mrmlScene.GetFirstNodeByName("CenterlinePoint")
    if centerlinePointNode:
      centerlinePointNode.SetNthControlPointPosition(0, 0, 0, 0)

    resultMatrix = vtk.vtkMatrix4x4()
    
    self.inputTransformSelector.currentNode().SetAndObserveMatrixTransformToParent(resultMatrix)
    # Set the initialization transform based on the first ground truth frame
    with open(f'{self.gtPathBox.text}/frame_data{str(self.stepCount)}.json', 'rb') as f:
      data = json.load(f)
      cameraPose = data['camera-pose']
      
      matrix = vtk.vtkMatrix4x4()
      matrix.SetElement(0, 0, cameraPose[0][0]); matrix.SetElement(0, 1, cameraPose[0][1]); matrix.SetElement(0, 2, cameraPose[0][2]); matrix.SetElement(0, 3, cameraPose[0][3])
      matrix.SetElement(1, 0, cameraPose[1][0]); matrix.SetElement(1, 1, cameraPose[1][1]); matrix.SetElement(1, 2, cameraPose[1][2]); matrix.SetElement(1, 3, cameraPose[1][3])
      matrix.SetElement(2, 0, cameraPose[2][0]); matrix.SetElement(2, 1, cameraPose[2][1]); matrix.SetElement(2, 2, cameraPose[2][2]); matrix.SetElement(2, 3, cameraPose[2][3])
      matrix.SetElement(3, 0, cameraPose[3][0]); matrix.SetElement(3, 1, cameraPose[3][1]); matrix.SetElement(3, 2, cameraPose[3][2]); matrix.SetElement(3, 3, cameraPose[3][3])
      startTransformNode = slicer.util.getNode("Start")
      startTransformNode.SetAndObserveMatrixTransformToParent(matrix)

    if self.pointCloudSelector.currentNode():
      self.createPointCloud(str(self.stepCount), None, True)


    if self.cameraAirwayPositionSelector.currentNode() and self.centerlineSelector.currentNode() and self.centerlineScalingCheckBox.isChecked():
      closestRadius = self.calculateClosestCenterlineRadius()
      self.centerlineStartRadius = closestRadius
      self.centerlineScaleFactor = closestRadius / self.centerlineStartRadius
      self.centerlineScaleLabel.text = f'{self.centerlineScaleFactor:.2f}'
    else:
      self.centerlineScaleFactor = 1
  
  def onMaskSelection(self, index):
    if index == 1:
      self.cornerCutBox.setEnabled(True)
      self.borderCutBox.setEnabled(True)
    else:
      self.cornerCutBox.setEnabled(False)
      self.borderCutBox.setEnabled(False)

  def onLoadPred(self):
    translationsFilename = ""
    eulerFilename = ""
    poseFilename = ""
    for filename in os.listdir(self.pathBox.text):
      if filename.startswith('translation') and filename.endswith('.npz'):
        translationsFilename = filename
      elif filename.startswith('euler') and filename.endswith('.npz'):
        eulerFilename = filename
      elif filename.startswith('pose') and filename.endswith('.npz'):
        poseFilename = filename
    if translationsFilename != "":
      self.translations_pred = np.load(f'{self.pathBox.text}/{translationsFilename}')
      print(f'Loaded translations; {self.translations_pred["a"].shape}')
    else:
      print(f'Missing translations')
    if eulerFilename != "":
      self.euler_angle_pred = np.load(f'{self.pathBox.text}/{eulerFilename}')
      print(f'Loaded angles: {self.euler_angle_pred["a"].shape}')
    else:
      print(f'Missing angles')
    if poseFilename != "":
      self.pose_pred = np.load(f'{self.pathBox.text}/{poseFilename}')
      print(f'Loaded poses: {self.pose_pred["a"].shape}')
    else:
      print(f'Missing poses')


  def createPointCloud(self, stepCountString, maskImage, usePointCloudReg = False):
    suffix = "_depth.npy"
    depthMapFilename = ""
    for filename in os.listdir(self.pathBox.text):
      if filename.endswith(suffix):
        stripped_filename = re.sub(suffix + '$', '', filename).lstrip('0').split('.')[0]
        if stripped_filename == stepCountString:
          depthMapFilename = filename
    depthMap = np.load(f'{self.pathBox.text}/{depthMapFilename}')[0][0]
    maskImage = None
    if self.maskSelectionComboBox.currentIndex == 0:
      prefix = "mask_"
      suffix = ".jpeg"
      maskFilename = ""
      maskFilename = f'{prefix}{stepCountString.zfill(10)}{suffix}'

      #maskImage = np.load(f'{self.pathBox.text}/{maskFilename}')[0][0]
      maskImage = cv2.imread(f'{self.pathBox.text}/{maskFilename}')
      maskImage = cv2.cvtColor(maskImage, cv2.COLOR_BGR2GRAY)
    if self.imageSelector.currentNode():
      self.depthMapToPointCloud(depthMap, slicer.util.arrayFromVolume(self.imageSelector.currentNode())[self.stepCount], maskImage, usePointCloudReg)
    else:
      self.depthMapToPointCloud(depthMap, None, maskImage, usePointCloudReg)

  def adjustSliceOffset(self):
    layoutManager = slicer.app.layoutManager()

    # Depth Map
    red = layoutManager.sliceWidget('Red')
    redLogic = red.sliceLogic()
    #redLogic.SetSliceOffset(self.stepCount//self.stepSkipBox.value - 1)
    redLogic.SetSliceOffset(self.stepCount-1)

    # RGB
    green = layoutManager.sliceWidget('Green')
    greenLogic = green.sliceLogic()
    greenLogic.SetSliceOffset(self.stepCount//self.stepSkipBox.value - 1)
    #greenLogic.SetSliceOffset(self.stepCount-1)
  
  def count_GT_files(self, directory):
    count = 0
    for filename in os.listdir(directory):
      if filename.startswith("frame_data") and filename.endswith(".json"):
        count += 1
    return count
  
  def onStepImage(self):
    maskImage = None
    self.stepCount = self.stepCount + self.stepSkipBox.value
    stepCountString = str(self.stepCount)
    
    #if self.stepCount > len(self.translations_pred['a']):
    if self.stepCount+self.stepSkipBox.value >= self.numberOfFrames:
      self.stepTimerButton.setChecked(False)
      self.onStepTimer()
      self.finished = True
      return
      
    self.stepLabel.setText(str(self.stepCount))
      
    # ------------------------------ Ground Truth ------------------------------
    if self.methodComboBox.currentText == "Ground Truth":
      if self.pointCloudSelector.currentNode():
        self.createPointCloud(stepCountString, maskImage)
          
      # Display image by moving slice offset
      self.adjustSliceOffset()

      # Start Load Ground Truth:
      prefix = "frame_data"
      frameDataFilename = ""
      frameDataFilename = f'frame_data{stepCountString}.json'
      
      # Grab pose data and use it
      with open(f'{self.gtPathBox.text}/{frameDataFilename}', 'rb') as f:
        data = json.load(f)
        cameraPose = data['camera-pose']
        
        matrix = vtk.vtkMatrix4x4()
        matrix.SetElement(0, 0, cameraPose[0][0]); matrix.SetElement(0, 1, cameraPose[0][1]); matrix.SetElement(0, 2, cameraPose[0][2]); matrix.SetElement(0, 3, cameraPose[0][3])
        matrix.SetElement(1, 0, cameraPose[1][0]); matrix.SetElement(1, 1, cameraPose[1][1]); matrix.SetElement(1, 2, cameraPose[1][2]); matrix.SetElement(1, 3, cameraPose[1][3])
        matrix.SetElement(2, 0, cameraPose[2][0]); matrix.SetElement(2, 1, cameraPose[2][1]); matrix.SetElement(2, 2, cameraPose[2][2]); matrix.SetElement(2, 3, cameraPose[2][3])
        matrix.SetElement(3, 0, cameraPose[3][0]); matrix.SetElement(3, 1, cameraPose[3][1]); matrix.SetElement(3, 2, cameraPose[3][2]); matrix.SetElement(3, 3, cameraPose[3][3])

        # Apply inverse of the initialization transform
        inverseParentMatrix = vtk.vtkMatrix4x4()
        #self.inputTransformSelector.currentNode().GetParentTransformNode().GetParentTransformNode().GetMatrixTransformToParent(inverseParentMatrix)
        self.inputTransformSelector.currentNode().GetParentTransformNode().GetMatrixTransformToWorld(inverseParentMatrix)
        inverseParentMatrix.Invert()
        matrix.Multiply4x4(inverseParentMatrix, matrix, matrix)
        
        self.inputTransformSelector.currentNode().SetAndObserveMatrixTransformToParent(matrix)
      
    # ------------------------------ None ------------------------------
    elif self.methodComboBox.currentText == "None":
      self.centerlineScaleFactor = 1.0
      if self.pointCloudSelector.currentNode():
        self.createPointCloud(stepCountString, maskImage)
          
      # Display image by moving slice offset
      self.adjustSliceOffset()
    # ------------------------------ ICP ------------------------------
    elif self.methodComboBox.currentText == "ICP Only":
      if self.pointCloudSelector.currentNode():
        self.createPointCloud(stepCountString, maskImage)
          
      # Display image by moving slice offset
      self.adjustSliceOffset()

      # # Apply past ICP transform to pose transform
      # parentMatrix = vtk.vtkMatrix4x4()
      # self.inputTransformSelector.currentNode().GetMatrixTransformToParent(parentMatrix)
      # childMatrix = vtk.vtkMatrix4x4()
      # self.icpTransformSelector.currentNode().GetMatrixTransformToParent(childMatrix)
      # combinedMatrix = vtk.vtkMatrix4x4()
      # combinedMatrix.Multiply4x4(parentMatrix, childMatrix, combinedMatrix)
      # self.inputTransformSelector.currentNode().SetMatrixTransformToParent(combinedMatrix)
      # identityMatrix = vtk.vtkMatrix4x4()
      # self.icpTransformSelector.currentNode().SetMatrixTransformToParent(identityMatrix)
      
      # Start ICP:
      # self.calculateAndSetICPTransform()
    # ------------------------------ Recorded Pose ------------------------------
    elif self.methodComboBox.currentText == "Recorded AI Pose":
      if self.pointCloudSelector.currentNode():
        self.createPointCloud(stepCountString, maskImage, True)
          
      # Display image by moving slice offset
      self.adjustSliceOffset()

      # Start AI Pose:
      scale = self.scaleSliderWidget.value
      rotationScale = self.rotationScaleSliderWidget.value

      previousMatrix = vtk.vtkMatrix4x4()
      self.inputTransformSelector.currentNode().GetMatrixTransformToParent(previousMatrix)
      step = ((self.stepCount-1-self.stepSkipBox.value) // self.stepSkipBox.value)+1
      pred_pose = self.get_transform(torch.from_numpy(self.euler_angle_pred['a'][step-1:step, 0]), torch.from_numpy(self.translations_pred['a'][step-1:step, 0]), scale, rotationScale)

      predMatrix = self.numpy_to_vtk_matrix(pred_pose)

      predMatrix.Invert()

      combinedMatrix = vtk.vtkMatrix4x4()
      combinedMatrix.Multiply4x4(previousMatrix, predMatrix, combinedMatrix)

      self.inputTransformSelector.currentNode().SetAndObserveMatrixTransformToParent(combinedMatrix)

    # ------------------------------ ICP with Pose ------------------------------
    elif self.methodComboBox.currentText == "Recorded AI Pose with Nudge":
      if self.pointCloudSelector.currentNode():
        self.createPointCloud(stepCountString, maskImage, True)
          
      # Display image by moving slice offset
      self.adjustSliceOffset()

      # Start AI Pose:
      scale = self.scaleSliderWidget.value
      rotationScale = self.rotationScaleSliderWidget.value

      previousMatrix = vtk.vtkMatrix4x4()
      self.inputTransformSelector.currentNode().GetMatrixTransformToParent(previousMatrix)
      step = ((self.stepCount-1-self.stepSkipBox.value) // self.stepSkipBox.value)+1
      pred_pose = self.get_transform(torch.from_numpy(self.euler_angle_pred['a'][step-1:step, 0]), torch.from_numpy(self.translations_pred['a'][step-1:step, 0]), scale, rotationScale)

      predMatrix = self.numpy_to_vtk_matrix(pred_pose)

      predMatrix.Invert()

      combinedMatrix = vtk.vtkMatrix4x4()
      combinedMatrix.Multiply4x4(previousMatrix, predMatrix, combinedMatrix)
      
      if ((self.stepCount-1) % self.nudgeInterval.value) == 0 and self.stepCount != 1:
        radiusCenterlineNode = self.centerlineSelector.currentNode()
        inverseParentMatrix = vtk.vtkMatrix4x4()
        self.inputTransformSelector.currentNode().GetParentTransformNode().GetMatrixTransformToWorld(inverseParentMatrix)
        combinedMatrix_world = vtk.vtkMatrix4x4()
        combinedMatrix_world.Multiply4x4(inverseParentMatrix, combinedMatrix, combinedMatrix_world)
        inverseParentMatrix.Invert()

        # Closest point on centerline
        closestPoint = self.findClosestPointOnLine(radiusCenterlineNode, [combinedMatrix_world.GetElement(0,3), combinedMatrix_world.GetElement(1,3), combinedMatrix_world.GetElement(2,3)])
        closestPoint = list(closestPoint) + [1]
        closestPointTransformed = [0,0,0,1]
        inverseParentMatrix.MultiplyPoint(closestPoint, closestPointTransformed) # in pose space

        # Determine if closest point is too far away
        centerlinePointDifference = abs(np.linalg.norm(np.array(closestPointTransformed[:3])) - np.linalg.norm(np.array(self.previousClosestPoint[:3])))
        if (centerlinePointDifference < self.centerlineNudgeThresholdWidget.value) or (self.previousClosestPoint == [np.nan, np.nan, np.nan, 1]):
          centerlinePointNode = slicer.mrmlScene.GetFirstNodeByName("CenterlinePoint")
          if centerlinePointNode:
            centerlinePointNode.SetNthControlPointPosition(0, closestPoint[0], closestPoint[1], closestPoint[2])

          nudgeVector = np.array([closestPointTransformed[0]-combinedMatrix.GetElement(0,3), closestPointTransformed[1]-combinedMatrix.GetElement(1,3), closestPointTransformed[2]-combinedMatrix.GetElement(2,3)])
          nudgeNorm = np.linalg.norm(nudgeVector)
          nudgeFactor = min(nudgeNorm, self.nudgeFactorWidget.value)
          nudgeVector = (nudgeVector/nudgeNorm) * nudgeFactor

          originalPoint = np.array([previousMatrix.GetElement(0,3), previousMatrix.GetElement(1,3), previousMatrix.GetElement(2,3)])
          originalVector = np.array([combinedMatrix.GetElement(0,3)-originalPoint[0], combinedMatrix.GetElement(1,3)-originalPoint[1], combinedMatrix.GetElement(2,3)-originalPoint[2]])
          originalNorm = np.linalg.norm(originalVector)

          nudgedPoint = [combinedMatrix.GetElement(0,3)+nudgeVector[0], combinedMatrix.GetElement(1,3)+nudgeVector[1], combinedMatrix.GetElement(2,3)+nudgeVector[2]]
          nudgedVector = np.array([nudgedPoint[0]-previousMatrix.GetElement(0,3), nudgedPoint[1]-previousMatrix.GetElement(1,3), nudgedPoint[2]-previousMatrix.GetElement(2,3)])
          nudgedNorm = np.linalg.norm(nudgedVector)

          adjustedVector = nudgedPoint-originalPoint
          adjustedNorm = np.linalg.norm(adjustedVector)
          adjustedVector = (adjustedVector/adjustedNorm) * originalNorm

          magnitudeDifference = originalNorm - nudgedNorm
          originalVector = (originalVector/originalNorm) * magnitudeDifference

          centerlineVector = np.array([0,0,0])
          # Re-orient to center line vector
          if self.previousClosestPoint != [np.nan, np.nan, np.nan, 1]:
            centerlineVector = np.array(closestPointTransformed) - np.array(self.previousClosestPoint)
            poseVector = [0,0,1,1]

            combinedRotationMatrix = self.getRotationMatrixfromMatrix(combinedMatrix)
            combinedRotationMatrix.MultiplyPoint(poseVector, poseVector)

            pointCloudRegMatrix = vtk.vtkMatrix4x4()
            pointCloudRegNode = slicer.util.getNode("PointCloudReg")
            pointCloudRegNode.GetMatrixTransformToParent(pointCloudRegMatrix)
            
            pointCloudRegMatrix.MultiplyPoint(poseVector, poseVector)
            poseVector = np.array(poseVector)

            if not np.array_equal(centerlineVector,[0.0,0.0,0.0,0.0]):
              centerlineRotationMatrix = self.numpy_to_vtk_matrix(self.partial_rotation_matrix_to_vector(poseVector[:3], centerlineVector[:3], self.nudgeRotationFactorWidget.value))
              centerlineRotationMatrix.Multiply4x4(combinedRotationMatrix, centerlineRotationMatrix, centerlineRotationMatrix)

              combinedMatrix.SetElement(0,0,centerlineRotationMatrix.GetElement(0,0)); combinedMatrix.SetElement(0,1,centerlineRotationMatrix.GetElement(0,1)); combinedMatrix.SetElement(0,2,centerlineRotationMatrix.GetElement(0,2))
              combinedMatrix.SetElement(1,0,centerlineRotationMatrix.GetElement(1,0)); combinedMatrix.SetElement(1,1,centerlineRotationMatrix.GetElement(1,1)); combinedMatrix.SetElement(1,2,centerlineRotationMatrix.GetElement(1,2))
              combinedMatrix.SetElement(2,0,centerlineRotationMatrix.GetElement(2,0)); combinedMatrix.SetElement(2,1,centerlineRotationMatrix.GetElement(2,1)); combinedMatrix.SetElement(2,2,centerlineRotationMatrix.GetElement(2,2))

          centerlineNorm = np.linalg.norm(centerlineVector)
          centerlineVector = (centerlineVector/centerlineNorm) * magnitudeDifference
          if centerlineNorm == 0:
            centerlineVector = np.array([0,0,0])
          combinedMatrix.SetElement(0,3,originalPoint[0]+nudgedVector[0]+centerlineVector[0])
          combinedMatrix.SetElement(1,3,originalPoint[1]+nudgedVector[1]+centerlineVector[1])
          combinedMatrix.SetElement(2,3,originalPoint[2]+nudgedVector[2]+centerlineVector[2])

          self.previousClosestPoint = closestPointTransformed

          self.nudgeLabel.text = f'Nudge by {nudgeFactor}'
        else:
          self.nudgeLabel.text = f'Center line point too far: {centerlinePointDifference}'
      else:
        self.nudgeLabel.text = ""

      self.inputTransformSelector.currentNode().SetAndObserveMatrixTransformToParent(combinedMatrix)

    # ------------------------------ ICP with Pose ------------------------------
    elif self.methodComboBox.currentText == "Recorded AI Pose with ICP":
      if self.pointCloudSelector.currentNode():
        self.createPointCloud(stepCountString, maskImage, True)
          
      # Display image by moving slice offset
      self.adjustSliceOffset()

      previousMatrix = vtk.vtkMatrix4x4()
      self.inputTransformSelector.currentNode().GetMatrixTransformToParent(previousMatrix)

      # Start ICP:
      if ((self.stepCount-1) % self.nudgeInterval.value) == 0 and self.stepCount != 1:
        icpMatrix = self.calculateICPTransform2()   

        combinedICPMatrix = vtk.vtkMatrix4x4()
        combinedICPMatrix.Multiply4x4(previousMatrix, icpMatrix, combinedICPMatrix)
        self.nudgeLabel.text = f'ICP performed'

      # Start AI Pose:
      scale = self.scaleSliderWidget.value
      rotationScale = self.rotationScaleSliderWidget.value

      step = ((self.stepCount-1-self.stepSkipBox.value) // self.stepSkipBox.value)+1
      pred_pose = self.get_transform(torch.from_numpy(self.euler_angle_pred['a'][step-1:step, 0]), torch.from_numpy(self.translations_pred['a'][step-1:step, 0]), scale, rotationScale)

      predMatrix = self.numpy_to_vtk_matrix(pred_pose)

      predMatrix.Invert()

      combinedMatrix = vtk.vtkMatrix4x4()
      combinedMatrix.Multiply4x4(combinedICPMatrix, predMatrix, combinedMatrix)

      self.inputTransformSelector.currentNode().SetAndObserveMatrixTransformToParent(combinedMatrix)
    # ------------------------------ ICP with Pose ------------------------------
    elif self.methodComboBox.currentText == "cGAN with ICP":
      if self.pointCloudSelector.currentNode():
        self.createPointCloud(stepCountString, maskImage)
          
      # Display image by moving slice offset
      self.adjustSliceOffset()

      # Start AI Pose:
      scale = self.scaleSliderWidget.value

      previousMatrix = vtk.vtkMatrix4x4()
      self.inputTransformSelector.currentNode().GetMatrixTransformToParent(previousMatrix)


      icpMatrix = self.calculateICPTransform2()
      #icpMatrix = self.remove_scaling(icpMatrix)

      combinedMatrix = vtk.vtkMatrix4x4()
      combinedMatrix.Multiply4x4(previousMatrix, icpMatrix, combinedMatrix)

      # Force it to move forwards a little bit
      icpVector = [0,0,1,1]
      rotationMatrix = self.getRotationMatrixfromMatrix(previousMatrix)
      rotationMatrix.MultiplyPoint(icpVector, icpVector)
      icpVector = np.array(icpVector[:3])
      icpVectorNorm = np.linalg.norm(icpVector)
      icpVector = (icpVector/icpVectorNorm) * self.icpcGANPushSlider.value
      combinedMatrix.SetElement(0,3,combinedMatrix.GetElement(0,3)+icpVector[0]); combinedMatrix.SetElement(1,3,combinedMatrix.GetElement(1,3)+icpVector[1]); combinedMatrix.SetElement(2,3,combinedMatrix.GetElement(2,3)+icpVector[2])

      self.inputTransformSelector.currentNode().SetAndObserveMatrixTransformToParent(combinedMatrix)

      self.nudgeLabel.text = f'ICP performed'

    # Calculates scale from centerline for the next step from the current step if there is a centerline
    if self.centerlineScalingCheckBox.isChecked():
      if self.cameraAirwayPositionSelector.currentNode() and self.centerlineSelector.currentNode():
        closestRadius = self.calculateClosestCenterlineRadius()
        self.centerlineScaleFactor = closestRadius / self.centerlineStartRadius
        self.centerlineScaleLabel.text = f'{self.centerlineScaleFactor:.2f}'
    else:
      self.centerlineScaleFactor = 1.0

  def remove_scaling(self, matrix):
    # Extract the scaling factors
    scale_x = np.linalg.norm([matrix.GetElement(0, 0), matrix.GetElement(1, 0), matrix.GetElement(2, 0)])
    scale_y = np.linalg.norm([matrix.GetElement(0, 1), matrix.GetElement(1, 1), matrix.GetElement(2, 1)])
    scale_z = np.linalg.norm([matrix.GetElement(0, 2), matrix.GetElement(1, 2), matrix.GetElement(2, 2)])
    
    # Normalize the columns
    for i in range(3):
      matrix.SetElement(i, 0, matrix.GetElement(i, 0) / scale_x)
      matrix.SetElement(i, 1, matrix.GetElement(i, 1) / scale_y)
      matrix.SetElement(i, 2, matrix.GetElement(i, 2) / scale_z)

    return matrix

  def get_transform(self, euler, translation, scale, rotationScale):
    eulerValues = [angle * rotationScale for angle in euler.cpu().numpy().squeeze()]
    #eulerValues = [angle for angle in euler.cpu().numpy().squeeze()]
    # the output of the network is in radians
    final_mat = np.eye(4)
    final_mat[:3,:3] = R.from_euler('zyx', eulerValues).as_matrix()
    #final_mat = final_mat * scale
    T = np.eye(4)
    T[:3,3] = (translation.cpu().numpy().squeeze()) * scale
    #T[:3,3] = (translation.cpu().numpy().squeeze())
    M = np.matmul(T, final_mat)
    return M

  def getRotationMatrixfromMatrix(self, matrix):
    rotationMatrix = vtk.vtkMatrix4x4()
    rotationMatrix.Identity()
    for j in range(3):
      for i in range(3):
        rotationMatrix.SetElement(i, j, matrix.GetElement(i, j))
    return rotationMatrix

  def partial_rotation_matrix_to_vector(self, vec1, vec2, max_angle_deg):
    """ Create a 4x4 transform matrix to partially rotate vec1 towards vec2 by a maximum angle """
    def rotation_matrix_from_axis_angle(axis, angle):
        """ Create a rotation matrix from an axis and angle using Rodrigues' rotation formula """
        axis = axis / np.linalg.norm(axis)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        one_minus_cos = 1 - cos_angle
        x, y, z = axis
        rotation_matrix = np.array([
            [cos_angle + x*x*one_minus_cos, x*y*one_minus_cos - z*sin_angle, x*z*one_minus_cos + y*sin_angle],
            [y*x*one_minus_cos + z*sin_angle, cos_angle + y*y*one_minus_cos, y*z*one_minus_cos - x*sin_angle],
            [z*x*one_minus_cos - y*sin_angle, z*y*one_minus_cos + x*sin_angle, cos_angle + z*z*one_minus_cos]
        ])
        return rotation_matrix

    # Normalize the vectors
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    # Calculate the full rotation axis and angle
    axis = np.cross(vec1, vec2)
    full_angle = np.arccos(np.dot(vec1, vec2))

    # Scale the angle to the desired partial angle
    max_angle_rad = np.deg2rad(max_angle_deg)
    partial_angle = min(full_angle, max_angle_rad)

    # Create the partial rotation matrix
    partial_rot_matrix = rotation_matrix_from_axis_angle(axis, partial_angle)

    # Create a 4x4 transform matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = partial_rot_matrix

    return transform_matrix

  def vtk_to_numpy_matrix(self, vtk_matrix):
    numpy_matrix = np.zeros((4, 4))
    for i in range(4):
      for j in range(4):
        numpy_matrix[i, j] = vtk_matrix.GetElement(i, j)
    return numpy_matrix
  
  def numpy_to_vtk_matrix(self, numpy_matrix):
    # Ensure the numpy matrix is 4x4
    assert numpy_matrix.shape == (4, 4), "The input numpy matrix must be 4x4"

    # Create an empty vtkMatrix4x4
    vtk_matrix = vtk.vtkMatrix4x4()

    # Fill the vtkMatrix4x4 with values from the numpy matrix
    for i in range(4):
      for j in range(4):
        vtk_matrix.SetElement(i, j, numpy_matrix[i, j])

    return vtk_matrix

  def dump(self, cam_to_world, source_to_target_transformations):
    Ms = []
    #cam_to_world = np.eye(4)
    Ms.append(cam_to_world)
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        Ms.append(cam_to_world)
    return Ms
  
  def MatrixFromRotation(self, angle, x, y, z, matrix):
    matrix.Identity()

    if (angle == 0.0 or (x == 0.0 and y == 0.0 and z == 0.0)):
      return

    # convert to radians
    angle = vtk.vtkMath.RadiansFromDegrees(angle)

    # make a normalized quaternion
    w = math.cos(0.5 * angle)
    f = math.sin(0.5 * angle) / math.sqrt(x * x + y * y + z * z)
    x = x * f
    y = y * f
    z = z * f

    # convert the quaternion to a matrix
    ww = w * w
    wx = w * x
    wy = w * y
    wz = w * z

    xx = x * x
    yy = y * y
    zz = z * z

    xy = x * y
    xz = x * z
    yz = y * z

    s = ww - xx - yy - zz

    matrix.SetElement(0,0, xx * 2 + s)
    matrix.SetElement(1,0, (xy + wz) * 2)
    matrix.SetElement(2,0, (xz - wy) * 2)

    matrix.SetElement(0,1, (xy - wz) * 2)
    matrix.SetElement(1,1, yy * 2 + s)
    matrix.SetElement(2,1, (yz + wx) * 2)

    matrix.SetElement(0,2, (xz + wy) * 2)
    matrix.SetElement(1,2, (yz - wx) * 2)
    matrix.SetElement(2,2, zz * 2 + s)

  def depthMapToPointCloud(self, depthImage, rgbImage, maskImage = None, usePointCloudReg = False):
    height = len(depthImage)
    width = len(depthImage[0])

    if maskImage is not None:
      # Mask
      #depthImage[~maskImage] = np.nan
      #depthImage[maskImage==0] = np.nan
      #depthImage = np.where(maskImage==[0,0,0], depthImage, np.nan)
      depthImage = depthImage.astype(float)
      depthImage[maskImage == 0] = np.nan
      #print(depthImage)
    else:
      # Cut corners and border
      cornerSize = self.cornerCutBox.value
      borderSize = self.borderCutBox.value
      depthImage[:cornerSize, :cornerSize] = np.nan
      depthImage[:cornerSize, -cornerSize:] = np.nan
      depthImage[-cornerSize:, :cornerSize] = np.nan
      depthImage[-cornerSize:, -cornerSize:] = np.nan
      depthImage[:borderSize, :] = np.nan
      depthImage[-borderSize:, :] = np.nan
      depthImage[:, :borderSize] = np.nan
      depthImage[:, -borderSize:] = np.nan

    minimumDepth = np.nanmin(depthImage)
    maximumDepth = np.nanmax(depthImage)
    zThreshold = minimumDepth + ((maximumDepth - minimumDepth) * ((self.thresholdBox.value)/100))

    points = vtk.vtkPoints()

    fx_d = self.focalLengthBox.value
    fy_d = self.focalLengthBox.value

    colorArray = None
    # if rgbImage.any():
    #   colorArray = vtk.util.numpy_support.numpy_to_vtk(rgbImage.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    #   colorArray.SetName("PointCloudColor")
    #   colorArray.SetNumberOfComponents(3)
    if rgbImage.any():
      colorArray = vtk.vtkUnsignedCharArray()
      colorArray.SetName("PointCloudColor")
      colorArray.SetNumberOfComponents(3)

    thresholdedColorArray = vtk.vtkUnsignedCharArray()
    for u in range(height):
      for v in range(width):
        vflipped = width - (v + 1)
        #vflipped = v
        z = depthImage[u][vflipped]
        if (not np.isnan(z)) and (z < zThreshold):
          world_x = (z * (u - (height/2)) / fx_d) * (self.imageScaleBox.value * self.centerlineScaleFactor)
          world_y = (z * (v - (width/2)) / fy_d) * (self.imageScaleBox.value * self.centerlineScaleFactor)
          world_z = z * self.depthScaleBox.value * self.centerlineScaleFactor
          points.InsertNextPoint(world_x, world_y, world_z)
          if colorArray:
            colorArray.InsertNextTuple3(rgbImage[u, vflipped][0], rgbImage[u, vflipped][1], rgbImage[u, vflipped][2])
        
    # Create junk polygons so that Slicer can actually display the point cloud
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    poly = vtk.vtkPolygon()
    poly.GetPointIds().SetNumberOfIds(points.GetNumberOfPoints())
    for n in range(points.GetNumberOfPoints()):
      poly.GetPointIds().SetId(n, n)
    polys = vtk.vtkCellArray()
    polys.InsertNextCell(poly)
    polydata.SetPolys(polys)

    if colorArray:
      polydata.GetPointData().SetScalars(colorArray)

    if usePointCloudReg:
      pointCloudRegTransform = vtk.vtkTransform()
      pointCloudRegMatrix = vtk.vtkMatrix4x4()
      pointCloudRegNode = slicer.util.getNode("PointCloudReg")
      pointCloudRegNode.GetMatrixTransformToParent(pointCloudRegMatrix)
      pointCloudRegTransform.SetMatrix(pointCloudRegMatrix)
      pointCloudRegTransformFilter = vtk.vtkTransformPolyDataFilter()
      pointCloudRegTransformFilter.SetInputData(polydata)
      pointCloudRegTransformFilter.SetTransform(pointCloudRegTransform)
      pointCloudRegTransformFilter.Update()
      polydata = pointCloudRegTransformFilter.GetOutput()      

    # Display the point cloud
    modelNode = self.pointCloudSelector.currentNode()

    modelNode.SetAndObservePolyData(polydata)
    modelDisplayNode = modelNode.GetModelDisplayNode()
    if modelDisplayNode is None:
      modelDisplayNode = slicer.vtkMRMLModelDisplayNode()
      modelDisplayNode.SetScene(slicer.mrmlScene)
      slicer.mrmlScene.AddNode(modelDisplayNode)
      modelNode.SetAndObserveDisplayNodeID(modelDisplayNode.GetID())
      modelDisplayNode.SetRepresentation(modelDisplayNode.PointsRepresentation)
      modelDisplayNode.SetPointSize(4)
      modelDisplayNode.SetOpacity(1.0)
      modelDisplayNode.SetColor(1,0,0)

  def onInitializationICP(self):
    icpMatrix = self.calculateICPTransform()
    initialICPNode = self.initializationICPTransformSelector.currentNode()

    # inverseTransformMatrix = vtk.vtkMatrix4x4()
    # initialICPNode.GetParentTransformNode().GetMatrixTransformToWorld(inverseTransformMatrix)
    # inverseTransformMatrix.Invert()
    # icpMatrix.Multiply4x4(inverseTransformMatrix, icpMatrix, icpMatrix)

    originalICPMatrix = vtk.vtkMatrix4x4()
    combinedICPMatrix = vtk.vtkMatrix4x4()
    initialICPNode.GetMatrixTransformToParent(originalICPMatrix)
    combinedICPMatrix.Multiply4x4(originalICPMatrix, icpMatrix, combinedICPMatrix)

    initialICPNode.SetAndObserveMatrixTransformToParent(combinedICPMatrix)

  def calculateICPTransform(self):
    #outputTrans = self.icpTransformSelector.currentNode()

    fixedModel = self.baseModelSelector.currentNode()
    movingModel = self.pointCloudSelector.currentNode()

    # Harden models
    hardenedFixedModelNode = slicer.vtkMRMLModelNode()
    slicer.mrmlScene.AddNode(hardenedFixedModelNode)
    fixedPolyData = vtk.vtkPolyData()
    fixedPolyData.DeepCopy(fixedModel.GetPolyData())
    hardenedFixedModelNode.SetAndObservePolyData(fixedPolyData)
    fixedTransformNode = fixedModel.GetParentTransformNode()
    if fixedTransformNode:
      hardenedFixedModelNode.SetAndObserveTransformNodeID(fixedTransformNode.GetID())
      hardenedFixedModelNode.HardenTransform()
      #fixedPolyData = hardenedFixedModelNode.GetPolyData()
    else:
      fixedPolyData = fixedModel.GetPolyData()

    hardenedMovingModelNode = slicer.vtkMRMLModelNode()
    slicer.mrmlScene.AddNode(hardenedMovingModelNode)
    movingPolyData = vtk.vtkPolyData()
    movingPolyData.DeepCopy(movingModel.GetPolyData())
    hardenedMovingModelNode.SetAndObservePolyData(movingPolyData)
    movingTransformNode = movingModel.GetParentTransformNode()
    if movingTransformNode:
      hardenedMovingModelNode.SetAndObserveTransformNodeID(movingTransformNode.GetID())
      hardenedMovingModelNode.HardenTransform()
      
      # Fix origin
      inverseTransformMatrix = vtk.vtkMatrix4x4()
      movingTransformNode.GetMatrixTransformToWorld(inverseTransformMatrix)
      inverseTransformMatrix.Invert()

      transform = vtk.vtkTransform()
      transform.SetMatrix(inverseTransformMatrix)

      transformFilter1 = vtk.vtkTransformPolyDataFilter()
      transformFilter1.SetInputData(movingPolyData)
      transformFilter1.SetTransform(transform)
      transformFilter1.Update()
      movingPolyData = transformFilter1.GetOutput()

      transformFilter2 = vtk.vtkTransformPolyDataFilter()
      transformFilter2.SetInputData(fixedPolyData)
      transformFilter2.SetTransform(transform)
      transformFilter2.Update()
      fixedPolyData = transformFilter2.GetOutput()

      #movingPolyData = hardenedMovingModelNode.GetPolyData()
    else:
      movingPolyData = movingModel.GetPolyData()

    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(movingPolyData)
    icp.SetTarget(fixedPolyData)
    if self.icpMethodComboBox.currentText == "Similarity":
      icp.GetLandmarkTransform().SetModeToSimilarity()
    elif self.icpMethodComboBox.currentText == "Rigid":
      icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMeanDistanceModeToAbsoluteValue()
    icp.SetMaximumNumberOfIterations(int(self.icpIterationsSlider.value))
    icp.SetMaximumMeanDistance(self.icpMaxDistanceSlider.value)
    icp.SetMaximumNumberOfLandmarks(int(self.icpLandmarksSlider.value))
    icp.SetCheckMeanDistance(int(self.checkMeanDistanceCheckBox.isChecked()))
    icp.StartByMatchingCentroidsOff()
    icp.Update()

    slicer.mrmlScene.RemoveNode(hardenedFixedModelNode)
    slicer.mrmlScene.RemoveNode(hardenedMovingModelNode)

    outputMatrix = vtk.vtkMatrix4x4()
    icp.GetMatrix(outputMatrix)
    return outputMatrix


  def calculateICPTransform2(self):
    #outputTrans = self.icpTransformSelector.currentNode()

    fixedModel = self.baseModelSelector.currentNode()
    movingModel = self.pointCloudSelector.currentNode()

    # Harden models
    # Move the fixed model into the initialized coordinate system
    fixedPolyData = vtk.vtkPolyData()
    fixedPolyData.DeepCopy(fixedModel.GetPolyData())

    inverseParentMatrix = vtk.vtkMatrix4x4()
    movingModel.GetParentTransformNode().GetMatrixTransformToWorld(inverseParentMatrix)

    # # Force it to move forwards a little bit
    # icpVector = [0,0,1,1]
    # rotationMatrix = self.getRotationMatrixfromMatrix(inverseParentMatrix)
    # rotationMatrix.MultiplyPoint(icpVector, icpVector)
    # icpVector = np.array(icpVector[:3])
    # icpVectorNorm = np.linalg.norm(icpVector)
    # icpVector = (icpVector/icpVectorNorm) * -50
    # print(icpVector)
    # inverseParentMatrix.SetElement(0,3,inverseParentMatrix.GetElement(0,3)+icpVector[0]); inverseParentMatrix.SetElement(1,3,inverseParentMatrix.GetElement(1,3)+icpVector[1]); inverseParentMatrix.SetElement(2,3,inverseParentMatrix.GetElement(2,3)+icpVector[2])

    #inverseParentMatrix = self.remove_scaling(inverseParentMatrix)
    inverseParentMatrix.Invert()
    inverseParentTransform = vtk.vtkTransform()
    inverseParentTransform.SetMatrix(inverseParentMatrix)
    inverseParentTransformFilter = vtk.vtkTransformPolyDataFilter()
    inverseParentTransformFilter.SetInputData(fixedPolyData)
    inverseParentTransformFilter.SetTransform(inverseParentTransform)
    inverseParentTransformFilter.Update()
    fixedPolyData = inverseParentTransformFilter.GetOutput()

    hardenedFixedModelNode = slicer.vtkMRMLModelNode()
    slicer.mrmlScene.AddNode(hardenedFixedModelNode)
    hardenedFixedModelNode.SetName("Fixed")
    hardenedFixedModelNode.SetAndObservePolyData(fixedPolyData)

    hardenedMovingModelNode = slicer.vtkMRMLModelNode()
    slicer.mrmlScene.AddNode(hardenedMovingModelNode)
    hardenedMovingModelNode.SetName("Moving")
    movingPolyData = vtk.vtkPolyData()
    movingPolyData.DeepCopy(movingModel.GetPolyData())
    hardenedMovingModelNode.SetAndObservePolyData(movingPolyData)

    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(movingPolyData)
    icp.SetTarget(fixedPolyData)
    if self.icpMethodComboBox.currentText == "Similarity":
      icp.GetLandmarkTransform().SetModeToSimilarity()
    elif self.icpMethodComboBox.currentText == "Rigid":
      icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMeanDistanceModeToAbsoluteValue()
    icp.SetMaximumNumberOfIterations(int(self.icpIterationsSlider.value))
    icp.SetMaximumMeanDistance(self.icpMaxDistanceSlider.value)
    icp.SetMaximumNumberOfLandmarks(int(self.icpLandmarksSlider.value))
    icp.SetCheckMeanDistance(int(self.checkMeanDistanceCheckBox.isChecked()))
    icp.StartByMatchingCentroidsOff()
    icp.Update()

    slicer.mrmlScene.RemoveNode(hardenedFixedModelNode)
    slicer.mrmlScene.RemoveNode(hardenedMovingModelNode)

    outputMatrix = vtk.vtkMatrix4x4()
    icp.GetMatrix(outputMatrix)
    return outputMatrix


  def calculateClosestCenterlineRadius(self):
    radiusCenterlineNode = self.centerlineSelector.currentNode()
    currentMatrix = vtk.vtkMatrix4x4()
    self.cameraAirwayPositionSelector.currentNode().GetMatrixTransformToWorld(currentMatrix)

    closestPoint, idx = self.findClosestPoint(radiusCenterlineNode, [currentMatrix.GetElement(0,3), currentMatrix.GetElement(1,3), currentMatrix.GetElement(2,3)])
    #print(closestPoint)
    closestRadius = (radiusCenterlineNode.GetPolyData().GetPointData().GetArray("Radius").GetValue(idx))

    return closestRadius

  def findClosestPoint(self, pathNode, point):
    pathPoints = pathNode.GetPolyData().GetPoints()
    closestPoint = pathPoints.GetPoint(0)
    closestDistance = vtk.vtkMath.Distance2BetweenPoints(point, closestPoint) #Distance is squared
    closestIdx = -1
    for idx in range(0, pathPoints.GetNumberOfPoints()):
      pathPoint = pathPoints.GetPoint(idx)
      distance = vtk.vtkMath.Distance2BetweenPoints(point, pathPoint)
      if distance <= closestDistance:
        closestDistance = distance
        closestPoint = pathPoint
        closestIdx = idx
    return closestPoint, closestIdx
  
  def findClosestPointOnLine(self, pathNode, point):
    path_pd = pathNode.GetPolyData()
    cellLocator = vtk.vtkCellLocator()
    cellLocator.SetDataSet(path_pd)
    cellLocator.BuildLocator()

    closestPoint = [0.0, 0.0, 0.0]  # To store the coordinates of the closest point
    cellId = vtk.mutable(0)  # To store the ID of the cell (line segment) containing the closest point
    subId = vtk.mutable(0)  # To store the ID of the sub-cell (not used for lines, so it will remain 0)
    dist2 = vtk.mutable(0.0)  # To store the squared distance from the query point to the closest point

    cellLocator.FindClosestPoint(point, closestPoint, cellId, subId, dist2)
    return closestPoint
  
class ReplayTrackerlessDataTest(ScriptedLoadableModuleTest):

  def setUp(self):
    widget = ReplayTrackerlessDataWidget()
    return widget
  
  # For generating curves
  def runTest(self):
    slicer.util.selectModule('CurveMaker')
    slicer.util.selectModule('CollectPoints')

    basePath = "G:/Partners HealthCare Dropbox/Franklin King/SNRLabDisk/Projects/CanonProj/TrackerlessNavigation/ExperimentResults/"

    self.delayDisplay("<h2>Clearing Previous Results</h2>")
    for fiducialListName in fiducialList:
      fiducialPoseOnlyNode = slicer.util.getNode(fiducialListName)
      fiducialPoseOnlyNode.RemoveAllControlPoints()

    self.delayDisplay("<h2>Starting Curves</h2>")
    for idx in range(len(dataPaths)):
      self.runCurveModule(dataPaths[idx], gtPaths[idx], stepSkips, scaleFactors, "Initial_ICP_I", fiducialList[idx], stepStartOffset)

    slicer.util.selectModule('ReplayTrackerlessData')    
  
  # # for replaying data
  # def runTest(self):
  #   slicer.util.selectModule('CurveMaker')
  #   slicer.util.selectModule('CollectPoints')

  #   basePath = "D:/Partners HealthCare Dropbox/Franklin King/SNRLabDisk/Projects/CanonProj/TrackerlessNavigation/ExperimentResults/"

  #   # Box Phantom 1
  #   dataPaths = [f"{basePath}Model_Results/BoxPhantom1/B_SelfSupervised-ArtifactRemoval-NoPoseLoss-LongtermLoss/Output/1/forward/frequency_3",
  #                f"{basePath}Model_Results/BoxPhantom1/B_SelfSupervised-ArtifactRemoval-NoPoseLoss-LongtermLoss/Output/2/forward/frequency_3",
  #                f"{basePath}Model_Results/BoxPhantom1/B_SelfSupervised-ArtifactRemoval-NoPoseLoss-LongtermLoss/Output/3/forward/frequency_3",
  #                f"{basePath}Model_Results/BoxPhantom1/B_SelfSupervised-ArtifactRemoval-NoPoseLoss-LongtermLoss/Output/4/forward/frequency_3"]
  #   gtPaths = [f"{basePath}Validation/BoxPhantom1/Images/1/forward",
  #              f"{basePath}Validation/BoxPhantom1/Images/2/forward",
  #              f"{basePath}Validation/BoxPhantom1/Images/3/forward",
  #              f"{basePath}Validation/BoxPhantom1/Images/4/forward"]
  #   stepSkips = [3, 3, 3, 3]
  #   scaleFactors = [22.85, 27.11, 20.49, 21.18]
  #   nudgeIntervals = [3, 3, 3, 3]
  #   nudgeFactors = [0.5, 0.5, 1.5, 1.5]
  #   nudgeRotationFactors = [0.5, 0.5, 1.5, 1.5]
  #   initial_ICPs = ["Initial_ICP_1", "Initial_ICP_2", "Initial_ICP_3", "Initial_ICP_4"]
  #   fiducialListPoseOnly = ["D_1_PoseOnly", "D_2_PoseOnly", "D_3_PoseOnly", "D_4_PoseOnly"]
  #   fiducialListCenterCorrection = ["D_1_CenterCorrection", "D_2_CenterCorrection", "D_3_CenterCorrection", "D_4_CenterCorrection"]
  #   stepStartOffset = 1

  #   # # Box Phantom 2
  #   # dataPaths = [f"{basePath}Model_Results/BoxPhantom2/D_SelfSupervised-ArtifactRemoval-PoseLoss-LongtermLoss/Output/1/forward/frequency_3",
  #   #              f"{basePath}Model_Results/BoxPhantom2/D_SelfSupervised-ArtifactRemoval-PoseLoss-LongtermLoss/Output/2/forward/frequency_3",
  #   #              f"{basePath}Model_Results/BoxPhantom2/D_SelfSupervised-ArtifactRemoval-PoseLoss-LongtermLoss/Output/3/forward/frequency_3",
  #   #              f"{basePath}Model_Results/BoxPhantom2/D_SelfSupervised-ArtifactRemoval-PoseLoss-LongtermLoss/Output/4/forward/frequency_3"]
  #   # gtPaths = [f"{basePath}Validation/BoxPhantom2/Images/1/forward",
  #   #            f"{basePath}Validation/BoxPhantom2/Images/2/forward",
  #   #            f"{basePath}Validation/BoxPhantom2/Images/3/forward",
  #   #            f"{basePath}Validation/BoxPhantom2/Images/4/forward"]
  #   # stepSkips = [3, 3, 3, 3]
  #   # scaleFactors = [15.60, 15.49, 24.95, 20.03]
  #   # nudgeIntervals = [3, 3, 3, 3]
  #   # nudgeFactors = [0.5, 0.5, 1.5, 1.5]
  #   # nudgeRotationFactors = [0.5, 0.5, 1.5, 1.5]
  #   # initial_ICPs = ["Initial_ICP_1", "Initial_ICP_2", "Initial_ICP_3", "Initial_ICP_4"]
  #   # fiducialListPoseOnly = ["D_1_PoseOnly", "D_2_PoseOnly", "D_3_PoseOnly", "D_4_PoseOnly"]
  #   # fiducialListCenterCorrection = ["D_1_CenterCorrection", "D_2_CenterCorrection", "D_3_CenterCorrection", "D_4_CenterCorrection"] 
  #   # stepStartOffset = 1

    # Full Phantom 2
    dataPaths = [f"{basePath}Model_Results/FullPhantom2/D_SelfSupervised-ArtifactRemoval-PoseLoss-LongtermLoss/Output/1/forward/frequency_3",
                 f"{basePath}Model_Results/FullPhantom2/D_SelfSupervised-ArtifactRemoval-PoseLoss-LongtermLoss/Output/2/forward/frequency_3",
                 f"{basePath}Model_Results/FullPhantom2/D_SelfSupervised-ArtifactRemoval-PoseLoss-LongtermLoss/Output/3/forward/frequency_3",
                 f"{basePath}Model_Results/FullPhantom2/D_SelfSupervised-ArtifactRemoval-PoseLoss-LongtermLoss/Output/4/forward/frequency_3",
                 f"{basePath}Model_Results/FullPhantom2/D_SelfSupervised-ArtifactRemoval-PoseLoss-LongtermLoss/Output/5/forward/frequency_3",
                 f"{basePath}Model_Results/FullPhantom2/D_SelfSupervised-ArtifactRemoval-PoseLoss-LongtermLoss/Output/6/forward/frequency_3",
                 f"{basePath}Model_Results/FullPhantom2/D_SelfSupervised-ArtifactRemoval-PoseLoss-LongtermLoss/Output/7/forward/frequency_3",
                 f"{basePath}Model_Results/FullPhantom2/D_SelfSupervised-ArtifactRemoval-PoseLoss-LongtermLoss/Output/8/forward/frequency_3"]
    gtPaths = [f"{basePath}Validation/FullPhantom2/Images/1/forward",
               f"{basePath}Validation/FullPhantom2/Images/2/forward",
               f"{basePath}Validation/FullPhantom2/Images/3/forward",
               f"{basePath}Validation/FullPhantom2/Images/4/forward",
               f"{basePath}Validation/FullPhantom2/Images/5/forward",
               f"{basePath}Validation/FullPhantom2/Images/6/forward",
               f"{basePath}Validation/FullPhantom2/Images/7/forward",
               f"{basePath}Validation/FullPhantom2/Images/8/forward"]
    stepSkips = [3, 3, 3, 3]
    scaleFactors = [18.33, 16.76, 24.75, 23.12, 17.02, 17.65, 20.48, 22.35]
    nudgeIntervals = [3, 3, 3, 3]
    nudgeFactors = [0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5]
    nudgeRotationFactors = [0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5]
    initial_ICPs = ["Initial_ICP_1", "Initial_ICP_2", "Initial_ICP_3", "Initial_ICP_4", "Initial_ICP_5", "Initial_ICP_6", "Initial_ICP_7", "Initial_ICP_8"]
    fiducialListPoseOnly = ["D_1_PoseOnly", "D_2_PoseOnly", "D_3_PoseOnly", "D_4_PoseOnly", "D_5_PoseOnly", "D_6_PoseOnly", "D_7_PoseOnly", "D_8_PoseOnly"]
    fiducialListCenterCorrection = ["D_1_CenterCorrection", "D_2_CenterCorrection", "D_3_CenterCorrection", "D_4_CenterCorrection", "D_5_CenterCorrection", "D_6_CenterCorrection", "D_7_CenterCorrection", "D_8_CenterCorrection"] 
    stepStartOffset = 100

  #   # # Rigid Phantom 1
  #   # dataPaths = [f"{basePath}Model_Results/RigidPhantom1/D_SelfSupervised-ArtifactRemoval-PoseLoss-LongtermLoss/Output/1/forward/frequency_3",
  #   #              f"{basePath}Model_Results/RigidPhantom1/D_SelfSupervised-ArtifactRemoval-PoseLoss-LongtermLoss/Output/3/forward/frequency_3",
  #   #              f"{basePath}Model_Results/RigidPhantom1/D_SelfSupervised-ArtifactRemoval-PoseLoss-LongtermLoss/Output/5/forward/frequency_3",
  #   #              f"{basePath}Model_Results/RigidPhantom1/D_SelfSupervised-ArtifactRemoval-PoseLoss-LongtermLoss/Output/7/forward/frequency_3"]
  #   # gtPaths = [f"{basePath}Validation/RigidPhantom1/Images/1/forward",
  #   #            f"{basePath}Validation/RigidPhantom1/Images/3/forward",
  #   #            f"{basePath}Validation/RigidPhantom1/Images/5/forward",
  #   #            f"{basePath}Validation/RigidPhantom1/Images/7/forward"]
  #   # stepSkips = [3, 3, 3, 3]
  #   # scaleFactors = [18.54, 14.63, 18.13, 19.84]
  #   # nudgeIntervals = [3, 3, 3, 3]
  #   # nudgeFactors = [0.5, 1.5, 0.5, 1.5]
  #   # nudgeRotationFactors = [0.5, 1.5, 0.5, 1.5]
  #   # initial_ICPs = ["Initial_ICP_1", "Initial_ICP_3", "Initial_ICP_5", "Initial_ICP_7"]
  #   # fiducialListPoseOnly = ["D_1_PoseOnly", "D_3_PoseOnly", "D_5_PoseOnly", "D_7_PoseOnly"]
  #   # fiducialListCenterCorrection = ["D_1_CenterCorrection", "D_3_CenterCorrection", "D_5_CenterCorrection", "D_7_CenterCorrection"]
  #   # stepStartOffset = 1


  #   self.delayDisplay("<h2>Clearing Previous Results</h2>")
  #   for fiducialListName in fiducialListPoseOnly:
  #     fiducialPoseOnlyNode = slicer.util.getNode(fiducialListName)
  #     fiducialPoseOnlyNode.RemoveAllControlPoints()
  #   for fiducialListName in fiducialListCenterCorrection:
  #     fiducialCenterCorrectionNode = slicer.util.getNode(fiducialListName)
  #     fiducialCenterCorrectionNode.RemoveAllControlPoints()

  #   self.delayDisplay("<h2>Starting Replays</h2>")
  #   for idx in range(len(dataPaths)):
  #     self.runReplayModule(dataPaths[idx], gtPaths[idx], stepSkips[idx], scaleFactors[idx], nudgeIntervals[idx], nudgeFactors[idx], nudgeRotationFactors[idx], initial_ICPs[idx], fiducialListPoseOnly[idx], fiducialListCenterCorrection[idx], stepStartOffset)

  #   # self.delayDisplay("<h2>Drawing Paths</h2>")
  #   # curveMakerWidget = slicer.modules.curvemaker.widgetRepresentation()
  #   # sourceWidget = curveMakerWidget.children()[1].children()[2]
  #   # curveWidget = curveMakerWidget.children()[1].children()[4]
  #   # radiusWidget = curveMakerWidget.children()[1].children()[6]
  #   # generateCurveButton = curveMakerWidget.children()[1].children()[18]
  #   # for fiducialListName in fiducialListPoseOnly:
  #   #   sourceWidget.setCurrentNode(slicer.util.getNode(fiducialListName))
  #   #   curveWidget.setCurrentNode(slicer.util.getNode(fiducialListName+"_Curve"))
  #   #   radiusWidget.value = 1
  #   #   self.wait_without_blocking(0.05)
  #   #   generateCurveButton.clicked()
  #   #   self.wait_without_blocking(0.05)
  #   # for fiducialListName in fiducialListCenterCorrection:
  #   #   sourceWidget.setCurrentNode(slicer.util.getNode(fiducialListName))
  #   #   curveWidget.setCurrentNode(slicer.util.getNode(fiducialListName+"_Curve"))
  #   #   radiusWidget.value = 1
  #   #   generateCurveButton.clicked()

  #   slicer.util.selectModule('ReplayTrackerlessData')

  def runReplayModule(self, dataPath, gtPath, stepSkip, scaleFactor, nudgeInterval, nudgeFactor, nudgeRotationFactor, initial_ICP_name, fiducialPoseOnly, fiducialCenterCorrection, stepStartOffset=1):
    widget = self.setUp()

    poseNode = slicer.util.getNode("Pose")
    initial_icp_node = slicer.util.getNode(initial_ICP_name)
    
    
    poseNode.SetAndObserveTransformNodeID(initial_icp_node.GetID())
    widget.pathBox.setText(dataPath)
    widget.gtPathBox.setText(gtPath)
    widget.stepSkipBox.value = stepSkip
    widget.stepStartOffsetBox.value = stepStartOffset
    widget.scaleSliderWidget.value = scaleFactor
    widget.nudgeInterval.value = nudgeInterval
    widget.nudgeFactorWidget.value = nudgeFactor
    widget.nudgeRotationFactorWidget.value = nudgeRotationFactor

    self.delayDisplay(f"<h1>Starting Replay: {fiducialPoseOnly}</h1>")
    fiducialPoseOnlyNode = None
    try:
      fiducialPoseOnlyNode = slicer.util.getNode(fiducialPoseOnly)
    except:
      fiducialPoseOnlyNode = None
    if fiducialPoseOnlyNode:
      widget.methodComboBox.setCurrentIndex(0)
      widget.onAutoInputs()
      widget.onLoadPred()
      widget.onResetStepCount()

      collectPointsWidget = slicer.modules.collectpoints.widgetRepresentation()
      collectPointsWidget.SamplingTransformNodeComboBox.setCurrentNodeID(poseNode.GetID())
      collectPointsWidget.OutputNodeComboBox.setCurrentNodeID(fiducialPoseOnlyNode.GetID())
      collectPointsWidget.CollectButton.clicked()

      while True:
        self.wait_without_blocking(0.01)
        widget.stepButton.clicked()
        self.wait_without_blocking(0.01)
        if widget.finished:
          break
        collectPointsWidget.CollectButton.clicked()

    self.delayDisplay(f"<h1>Starting Replay: {fiducialCenterCorrection}</h1>")
    fiducialCenterCorrectionNode = None
    try:
      fiducialCenterCorrectionNode = slicer.util.getNode(fiducialCenterCorrection)
    except:
      fiducialCenterCorrectionNode = None
    if fiducialCenterCorrectionNode:
      widget.methodComboBox.setCurrentIndex(1)
      widget.onAutoInputs()
      widget.onLoadPred()
      widget.onResetStepCount()

      collectPointsWidget = slicer.modules.collectpoints.widgetRepresentation()
      collectPointsWidget.SamplingTransformNodeComboBox.setCurrentNodeID(poseNode.GetID())
      collectPointsWidget.OutputNodeComboBox.setCurrentNodeID(fiducialCenterCorrectionNode.GetID())
      collectPointsWidget.CollectButton.clicked()

      while True:
        self.wait_without_blocking(0.01)
        widget.stepButton.clicked()
        self.wait_without_blocking(0.01)
        if widget.finished:
          break
        collectPointsWidget.CollectButton.clicked()    

    widget.cleanup()
    widget.parent.deleteLater()      

  def wait_without_blocking(self, timeout=10):
    start_time = time.time()
    while True:
      slicer.app.processEvents()
      if time.time() - start_time > timeout:
        break

  def runCurveModule(self, dataPath, gtPath, stepSkip, scaleFactor, initial_ICP_name, fiducialPoseOnly, stepStartOffset=1):
    widget = self.setUp()

    poseNode = slicer.util.getNode("Pose")
    initial_icp_node = slicer.util.getNode(initial_ICP_name)
    
    poseNode.SetAndObserveTransformNodeID(initial_icp_node.GetID())
    widget.pathBox.setText(dataPath)
    widget.gtPathBox.setText(gtPath)
    widget.stepSkipBox.value = stepSkip
    widget.scaleSliderWidget.value = scaleFactor
    widget.stepStartOffsetBox.value = stepStartOffset

    self.delayDisplay(f"<h1>Starting Replay: {fiducialPoseOnly}</h1>")
    fiducialPoseOnlyNode = None
    try:
      fiducialPoseOnlyNode = slicer.util.getNode(fiducialPoseOnly)
    except:
      fiducialPoseOnlyNode = None
    if fiducialPoseOnlyNode:
      widget.methodComboBox.setCurrentIndex(0)
      widget.onAutoInputs()
      widget.onLoadPred()
      widget.onResetStepCount()

      collectPointsWidget = slicer.modules.collectpoints.widgetRepresentation()
      collectPointsWidget.SamplingTransformNodeComboBox.setCurrentNodeID(poseNode.GetID())
      collectPointsWidget.OutputNodeComboBox.setCurrentNodeID(fiducialPoseOnlyNode.GetID())
      collectPointsWidget.CollectButton.clicked()

      while True:
        self.wait_without_blocking(0.01)
        widget.stepButton.clicked()
        self.wait_without_blocking(0.01)
        if widget.finished:
          break
        collectPointsWidget.CollectButton.clicked()

    widget.cleanup()
    widget.parent.deleteLater()


