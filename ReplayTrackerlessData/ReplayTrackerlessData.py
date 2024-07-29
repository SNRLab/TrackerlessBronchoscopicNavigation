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

  def cleanup(self):
    self.stepTimer.stop()
    self.centerlineScaleFactor = 1.0
    self.centerlineStartRadius = 1.0
    self.onResetStepCount()

  def onReload(self,moduleName="ReplayTrackerlessData"):
    self.stepTimer.stop()
    self.centerlineScaleFactor = 1.0
    self.centerlineStartRadius = 1.0
    self.onResetStepCount()
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
    self.gtPathBox = qt.QLineEdit("D:/MeghaData/Data/28_06_2024/depth_gauss_mask_min_pose_longterm_consistency_0.0001/data")
    self.gtPathBox.setReadOnly(True)
    self.gtPathButton = qt.QPushButton("...")
    self.gtPathButton.clicked.connect(self.select_directory_gt)
    gtPathBoxLayout = qt.QHBoxLayout()
    gtPathBoxLayout.addWidget(self.gtPathBox)
    gtPathBoxLayout.addWidget(self.gtPathButton)
    stepModeLayout.addRow("Data: ", gtPathBoxLayout)

    self.modelPathBox = qt.QLineEdit("D:/MeghaData/Data/28_06_2024/depth_gauss_mask_min_pose_longterm_consistency_0.0001/model")
    self.modelPathBox.setReadOnly(True)
    self.modelPathButton = qt.QPushButton("...")
    self.modelPathButton.clicked.connect(self.select_directory_model)
    modelPathBoxLayout = qt.QHBoxLayout()
    modelPathBoxLayout.addWidget(self.modelPathBox)
    modelPathBoxLayout.addWidget(self.modelPathButton)
    stepModeLayout.addRow("Model: ", modelPathBoxLayout)

    self.methodComboBox = qt.QComboBox()
    self.methodComboBox.addItem('ICP Only')
    self.methodComboBox.addItem('Recorded AI Pose')
    self.methodComboBox.addItem('Recorded AI Pose with Nudge')
    self.methodComboBox.addItem('AI Pose Inference')
    self.methodComboBox.addItem('AI Pose + ICP')
    self.methodComboBox.addItem('Ground Truth')
    self.methodComboBox.addItem('None')
    self.methodComboBox.setCurrentIndex(1)
    stepModeLayout.addRow('Registration Method:', self.methodComboBox)

    # self.gtCheckBox = qt.QCheckBox("Use Ground Truth Data")
    # self.gtCheckBox.setChecked(False)
    # stepModeLayout.addRow(self.gtCheckBox)

    # self.icpCheckBox = qt.QCheckBox("Use ICP")
    # self.icpCheckBox.setChecked(False)
    # stepModeLayout.addRow(self.icpCheckBox)

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
    self.stepFPSBox.value = 10
    stepModeLayout.addRow(self.stepFPSBox)

    self.scaleSliderWidget = ctk.ctkSliderWidget()
    self.scaleSliderWidget.setDecimals(2)
    self.scaleSliderWidget.minimum = 0.00
    self.scaleSliderWidget.maximum = 10000.00
    self.scaleSliderWidget.singleStep = 0.01
    self.scaleSliderWidget.value = 200.00
    stepModeLayout.addRow("Scale Factor:", self.scaleSliderWidget)

    # # Model Mode collapsible button
    # self.modelModeCollapsibleButton = ctk.ctkCollapsibleButton()
    # self.modelModeCollapsibleButton.text = "Model mode - Live data"
    # self.modelModeCollapsibleButton.collapsed = False
    # self.layout.addWidget(self.modelModeCollapsibleButton)
    # stepModeLayout = qt.QFormLayout(self.modelModeCollapsibleButton)

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
    self.pointCloudSelector.noneEnabled = False
    self.pointCloudSelector.showHidden = False
    self.pointCloudSelector.showChildNodeTypes = False
    self.pointCloudSelector.setMRMLScene(slicer.mrmlScene)
    depthMapModeLayout.addRow("Point Cloud Model: ", self.pointCloudSelector)

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
    self.maskSelectionComboBox.setCurrentIndex(0)
    self.maskSelectionComboBox.currentIndexChanged.connect(self.onMaskSelection)
    depthMapModeLayout.addRow('Mask method:', self.maskSelectionComboBox)

    self.cornerCutBox = qt.QSpinBox()
    self.cornerCutBox.setSingleStep(1)
    self.cornerCutBox.setMaximum(100)
    self.cornerCutBox.setMinimum(0)
    self.cornerCutBox.setSuffix(" px")
    self.cornerCutBox.value = 40
    self.cornerCutBox.setEnabled(False)
    depthMapModeLayout.addRow("Corner size to cut:", self.cornerCutBox)

    self.borderCutBox = qt.QSpinBox()
    self.borderCutBox.setSingleStep(1)
    self.borderCutBox.setMaximum(100)
    self.borderCutBox.setMinimum(0)
    self.borderCutBox.setSuffix(" px")
    self.borderCutBox.value = 40
    self.borderCutBox.setEnabled(False)
    depthMapModeLayout.addRow("Border size to cut:", self.borderCutBox)    

    self.focalLengthBox = ctk.ctkDoubleSpinBox()
    self.focalLengthBox.maximum = 100000.0
    self.focalLengthBox.minimum = 0.0
    self.focalLengthBox.setDecimals(6)
    self.focalLengthBox.setValue(128.7)
    depthMapModeLayout.addRow("Focal Length:", self.focalLengthBox)

    self.imageScaleBox = ctk.ctkDoubleSpinBox()
    self.imageScaleBox.setValue(17.0)
    self.imageScaleBox.maximum = 100000.0
    self.imageScaleBox.minimum = 0.0
    depthMapModeLayout.addRow("Base Image Scale: ", self.imageScaleBox)

    self.depthScaleBox = ctk.ctkDoubleSpinBox()
    self.depthScaleBox.setValue(17.0)
    self.depthScaleBox.maximum = 100000.0
    self.depthScaleBox.minimum = 0.0
    depthMapModeLayout.addRow("Base Depth Scale: ", self.depthScaleBox)

    self.thresholdBox = qt.QSpinBox()
    self.thresholdBox.setSingleStep(1)
    self.thresholdBox.setMaximum(100)
    self.thresholdBox.setMinimum(0)
    self.thresholdBox.setSuffix("%")
    self.thresholdBox.value = 60
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

    self.icpTransformSelector = slicer.qMRMLNodeComboBox()
    self.icpTransformSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.icpTransformSelector.selectNodeUponCreation = False
    self.icpTransformSelector.noneEnabled = False
    self.icpTransformSelector.addEnabled = True
    self.icpTransformSelector.removeEnabled = True
    self.icpTransformSelector.setMRMLScene(slicer.mrmlScene)
    icpLayout.addRow("ICP Transform: ", self.icpTransformSelector)

    self.checkMeanDistanceCheckBox = qt.QCheckBox("Check Mean Distance")
    self.checkMeanDistanceCheckBox.setChecked(True)
    icpLayout.addRow(self.checkMeanDistanceCheckBox)

    self.icpIterationsSlider = ctk.ctkSliderWidget()
    self.icpIterationsSlider.decimals = 0
    self.icpIterationsSlider.singleStep = 1
    self.icpIterationsSlider.minimum = 1
    self.icpIterationsSlider.maximum = 10000
    self.icpIterationsSlider.value = 2000
    icpLayout.addRow("Number of Iterations:", self.icpIterationsSlider)

    self.icpLandmarksSlider = ctk.ctkSliderWidget()
    self.icpLandmarksSlider.decimals = 0
    self.icpLandmarksSlider.singleStep = 1
    self.icpLandmarksSlider.minimum = 1
    self.icpLandmarksSlider.maximum = 2000
    self.icpLandmarksSlider.value = 200
    icpLayout.addRow("Number of Landmarks:", self.icpLandmarksSlider)

    self.icpMaxDistanceSlider = ctk.ctkSliderWidget()
    self.icpMaxDistanceSlider.decimals = 5
    self.icpMaxDistanceSlider.singleStep = 0.00001
    self.icpMaxDistanceSlider.minimum = 0.00001
    self.icpMaxDistanceSlider.maximum = 1.0
    self.icpMaxDistanceSlider.value = 0.001
    icpLayout.addRow("Maximum of Distance:", self.icpMaxDistanceSlider)

    # Add vertical spacer
    self.layout.addStretch(1)

  def onAutoInputs(self):
    self.imageSelector.setCurrentNode(slicer.util.getNode('0000000001'))
    self.inputTransformSelector.setCurrentNode(slicer.util.getNode('Pose'))
    self.pointCloudSelector.setCurrentNode(slicer.util.getNode('PointCloud'))
    self.centerlineSelector.setCurrentNode(slicer.util.getNode('CenterlineModel'))
    self.cameraAirwayPositionSelector.setCurrentNode(slicer.util.getNode('AirwayPosition'))
    self.baseModelSelector.setCurrentNode(slicer.util.getNode('airway'))
    self.icpTransformSelector.setCurrentNode(slicer.util.getNode('ICPTransform'))

  def select_directory_gt(self):
    directory = qt.QFileDialog.getExistingDirectory(self.parent, "Select Directory")
    if directory:
      self.gtPathBox.setText(directory)

  def select_directory_model(self):
    directory = qt.QFileDialog.getExistingDirectory(self.parent, "Select Directory")
    if directory:
      self.modelPathBox.setText(directory)

  def onStepTimer(self):
    if self.stepTimerButton.isChecked():
      self.stepTimer.start(int(1000/int(self.stepFPSBox.value)))
      self.stepFPSBox.enabled = False
    else:
      self.stepTimer.stop()
      self.stepFPSBox.enabled = True

  def onResetStepCount(self):
    self.stepCount = 1
    self.stepLabel.setText('1')

    self.centerlineScaleFactor = 1.0

    layoutManager = slicer.app.layoutManager()
    red = layoutManager.sliceWidget('Red')
    redLogic = red.sliceLogic()
    redLogic.SetSliceOffset(0)

    green = layoutManager.sliceWidget('Green')
    greenLogic = green.sliceLogic()
    greenLogic.SetSliceOffset(0)
    
    resultMatrix = vtk.vtkMatrix4x4()
    if self.inputTransformSelector.currentNode():
      self.inputTransformSelector.currentNode().SetAndObserveMatrixTransformToParent(resultMatrix)
    icpMatrix = vtk.vtkMatrix4x4()
    if self.icpTransformSelector.currentNode():
      self.icpTransformSelector.currentNode().SetAndObserveMatrixTransformToParent(icpMatrix)
  
  def onMaskSelection(self, index):
    if index == 1:
      self.cornerCutBox.setEnabled(True)
      self.borderCutBox.setEnabled(True)
    else:
      self.cornerCutBox.setEnabled(False)
      self.borderCutBox.setEnabled(False)

  def onLoadPred(self):
    self.translations_pred = np.load(f'{self.gtPathBox.text}/translation.npz')
    self.euler_angle_pred = np.load(f'{self.gtPathBox.text}/eulerangle.npz')
    self.pose_pred = np.load(f'{self.gtPathBox.text}/pose_prediction.npz')

  def createPointCloud(self, stepCountString, maskImage):
    suffix = "_depth.npy"
    depthMapFilename = ""
    for filename in os.listdir(self.gtPathBox.text):
      if filename.endswith(suffix):
        stripped_filename = re.sub(suffix + '$', '', filename).lstrip('0').split('.')[0]
        if stripped_filename == stepCountString:
          depthMapFilename = filename
    depthMap = np.load(f'{self.gtPathBox.text}/{depthMapFilename}')[0][0]

    if self.maskSelectionComboBox.currentIndex == 1:
      prefix = "mask_"
      suffix = ".jpeg"
      maskFilename = ""
      for filename in os.listdir(self.gtPathBox.text):
        if filename.startswith(prefix) and filename.endswith(suffix):
          base_name = filename.split(".")[0]
          number_str = base_name.split("_")[1]
          if number_str == stepCountString:
            maskFilename = filename
      maskImage = np.load(f'{self.gtPathBox.text}/{maskFilename}')[0][0]
    if self.imageSelector.currentNode():
      self.depthMapToPointCloud(depthMap, slicer.util.arrayFromVolume(self.imageSelector.currentNode())[self.stepCount], maskImage)
    else:
      self.depthMapToPointCloud(depthMap, None, maskImage)

  def adjustSliceOffset(self):
    layoutManager = slicer.app.layoutManager()

    # Depth Map
    red = layoutManager.sliceWidget('Red')
    redLogic = red.sliceLogic()
    redLogic.SetSliceOffset(self.stepCount//self.stepSkipBox.value - 1)

    # RGB
    green = layoutManager.sliceWidget('Green')
    greenLogic = green.sliceLogic()
    greenLogic.SetSliceOffset(self.stepCount - 1)

  def onStepImage(self):
    stepCountString = str(self.stepCount)
    maskImage = None

    # Calculates scale from centerline for the next step from the current step if there is a centerline
    if self.cameraAirwayPositionSelector.currentNode() and self.centerlineSelector.currentNode():
      closestRadius = self.calculateClosestCenterlineRadius()
      if self.stepCount <= 1:
        self.centerlineStartRadius = closestRadius
      self.centerlineScaleFactor = closestRadius / self.centerlineStartRadius
      self.centerlineScaleLabel.text = f'{self.centerlineScaleFactor:.2f}'
    # ------------------------------ Ground Truth ------------------------------
    if self.methodComboBox.currentText == "Ground Truth":
      if self.pointCloudSelector.currentNode():
        self.createPointCloud(stepCountString, maskImage)
          
      # Display image by moving slice offset
      self.adjustSliceOffset()

      # Start Load Ground Truth:

      prefix = "frame_data"
      frameDataFilename = ""
      for filename in os.listdir(self.gtPathBox.text):
        if filename.startswith(prefix):
          stripped_filename = re.sub('^' + prefix, '', filename).lstrip('0').split('.')[0]
          if stripped_filename == stepCountString:
            frameDataFilename = filename

      # Grab pose data and use it
      f = open(f'{self.gtPathBox.text}/{frameDataFilename}')
      data = json.load(f)
      cameraPose = data['camera-pose']
      
      matrix = vtk.vtkMatrix4x4()
      matrix.SetElement(0, 0, cameraPose[0][0]); matrix.SetElement(0, 1, cameraPose[0][1]); matrix.SetElement(0, 2, cameraPose[0][2]); matrix.SetElement(0, 3, cameraPose[0][3])
      matrix.SetElement(1, 0, cameraPose[1][0]); matrix.SetElement(1, 1, cameraPose[1][1]); matrix.SetElement(1, 2, cameraPose[1][2]); matrix.SetElement(1, 3, cameraPose[1][3])
      matrix.SetElement(2, 0, cameraPose[2][0]); matrix.SetElement(2, 1, cameraPose[2][1]); matrix.SetElement(2, 2, cameraPose[2][2]); matrix.SetElement(2, 3, cameraPose[2][3])
      matrix.SetElement(3, 0, cameraPose[3][0]); matrix.SetElement(3, 1, cameraPose[3][1]); matrix.SetElement(3, 2, cameraPose[3][2]); matrix.SetElement(3, 3, cameraPose[3][3])
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

      # Apply past ICP transform to pose transform
      parentMatrix = vtk.vtkMatrix4x4()
      self.inputTransformSelector.currentNode().GetMatrixTransformToParent(parentMatrix)
      childMatrix = vtk.vtkMatrix4x4()
      self.icpTransformSelector.currentNode().GetMatrixTransformToParent(childMatrix)
      combinedMatrix = vtk.vtkMatrix4x4()
      combinedMatrix.Multiply4x4(parentMatrix, childMatrix, combinedMatrix)
      self.inputTransformSelector.currentNode().SetMatrixTransformToParent(combinedMatrix)
      identityMatrix = vtk.vtkMatrix4x4()
      self.icpTransformSelector.currentNode().SetMatrixTransformToParent(identityMatrix)
      
      # Start ICP:
      self.calculateAndSetICPTransform()
    # ------------------------------ Recorded Pose ------------------------------
    elif self.methodComboBox.currentText == "Recorded AI Pose":
      if self.pointCloudSelector.currentNode():
        self.createPointCloud(stepCountString, maskImage)
          
      # Display image by moving slice offset
      self.adjustSliceOffset()

      # Start AI Pose:
      scale = self.scaleSliderWidget.value

      previousMatrix = vtk.vtkMatrix4x4()
      self.inputTransformSelector.currentNode().GetMatrixTransformToParent(previousMatrix)

      pred_pose = self.get_transform(torch.from_numpy(self.euler_angle_pred['a'][self.stepCount-1:self.stepCount, 0]), torch.from_numpy(self.translations_pred['a'][self.stepCount-1:self.stepCount, 0]), scale)

      combinedMatrix = vtk.vtkMatrix4x4()
      combinedMatrix.Multiply4x4(previousMatrix, self.numpy_to_vtk_matrix(pred_pose), combinedMatrix)

      self.inputTransformSelector.currentNode().SetAndObserveMatrixTransformToParent(combinedMatrix)

    # ------------------------------ ICP with Pose ------------------------------
    elif self.methodComboBox.currentText == "Recorded AI Pose with Nudge":
      if self.pointCloudSelector.currentNode():
        self.createPointCloud(stepCountString, maskImage)
          
      # Display image by moving slice offset
      self.adjustSliceOffset()

      # Start AI Pose:
      scale = self.scaleSliderWidget.value

      previousMatrix = vtk.vtkMatrix4x4()
      self.inputTransformSelector.currentNode().GetMatrixTransformToParent(previousMatrix)

      pred_pose = self.get_transform(torch.from_numpy(self.euler_angle_pred['a'][self.stepCount-1:self.stepCount, 0]), torch.from_numpy(self.translations_pred['a'][self.stepCount-1:self.stepCount, 0]), scale)

      combinedMatrix = vtk.vtkMatrix4x4()
      combinedMatrix.Multiply4x4(previousMatrix, self.numpy_to_vtk_matrix(pred_pose), combinedMatrix)

      self.inputTransformSelector.currentNode().SetAndObserveMatrixTransformToParent(combinedMatrix)

    # ------------------------------ Model ------------------------------
    elif self.methodComboBox.currentText == "AI Pose + ICP":
      pass
      # if self.pointCloudSelector.currentNode():
      #   self.createPointCloud(stepCountString, maskImage)

      # # Display image by moving slice offset
      # self.adjustSliceOffset()

      # scale = self.scaleSliderWidget.value

      # previousMatrix = vtk.vtkMatrix4x4()
      # self.inputTransformSelector.currentNode().GetMatrixTransformToParent(previousMatrix)

      # pred_poses = []
      # pred_poses.append(layers.transformation_from_parameters(torch.from_numpy(self.euler_angle_pred['a'][self.stepCount-1:self.stepCount, 0]), torch.from_numpy(self.translations_pred['a'][self.stepCount-1:self.stepCount, 0]) * scale).cpu().numpy())
      # pred_poses = np.concatenate(pred_poses)
      # dump_our = np.array(self.dump(self.vtk_to_numpy_matrix(previousMatrix), pred_poses))

      # self.inputTransformSelector.currentNode().SetAndObserveMatrixTransformToParent(self.numpy_to_vtk_matrix(dump_our[1]))

    self.stepCount = self.stepCount + self.stepSkipBox.value
    self.stepLabel.setText(stepCountString)

  def get_transform(self, euler, translation, scale):
    # the output of the network is in radians
    final_mat = np.eye(4)
    final_mat[:3,:3] = R.from_euler('zyx', euler.cpu().numpy().squeeze()).as_matrix()
    T = np.eye(4)
    T[:3,3] = (translation.cpu().numpy().squeeze()) * scale
    M = np.matmul(T, final_mat)
    return M

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
  
  def registerToBaseModel(self):
    # ICP Registration
    self.surfaceRegistration.inputFixedModelSelector.currentNodeID = self.baseModelSelector.currentNodeID
    
    self.surfaceRegistration.inputMovingModelSelector.currentNodeID = self.pointCloudSelector.currentNodeID
    self.surfaceRegistration.outputTransformSelector.currentNodeID = self.transientTransformSelector.currentNodeID
    #self.surfaceRegistration.landmarkTransformTypeButtonsSimilarity.checked = True
    self.surfaceRegistration.numberOfIterations.setValue(1000)
    self.surfaceRegistration.numberOfLandmarks.setValue(250)
    self.surfaceRegistration.onComputeButton()

    # # Add transient transform to total transform
    # transientMatrix = vtk.vtkMatrix4x4()
    # matrix = vtk.vtkMatrix4x4()
    # resultMatrix = vtk.vtkMatrix4x4()
    # self.transformSelector.currentNode().GetMatrixTransformToParent(matrix)
    # self.transientTransformSelector.currentNode().GetMatrixTransformToParent(transientMatrix)
    # vtk.vtkMatrix4x4.Multiply4x4(transientMatrix, matrix, resultMatrix)
    # self.transformSelector.currentNode().SetMatrixTransformToParent(resultMatrix)

  def depthMapToPointCloud(self, depthImage, rgbImage, maskImage = None):
    height = len(depthImage)
    width = len(depthImage[0])

    if maskImage:
      # Mask
      depthImage[~maskImage] = np.nan
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
        z = depthImage[u][vflipped]
        #z = depthImage[u][v]

        if (z is not np.nan) and (z < zThreshold):
          world_x = (z * (u - (height/2)) / fx_d) * (self.imageScaleBox.value * self.centerlineScaleFactor)
          world_y = (z * (v - (width/2)) / fy_d) * (self.imageScaleBox.value * self.centerlineScaleFactor)
          world_z = z * self.depthScaleBox.value * self.centerlineScaleFactor
          points.InsertNextPoint(world_x, world_y, world_z)
          #points.InsertNextPoint(np.array([z * (u - (height/2)) / fx_d, z * (v - (width/2)) / fy_d, z / self.depthDividerBox.value]))
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

    # # Temp
    # modelNodePerm = slicer.vtkMRMLModelNode() # Create new model node; temporary for showing all creations
    # slicer.mrmlScene.AddNode(modelNodePerm)
    # transformNode = self.inputTransformSelector.currentNode()
    # modelNodePerm.SetAndObserveTransformNodeID(transformNode.GetID())
    # slicer.vtkSlicerTransformLogic().hardenTransform(modelNodePerm)
    # modelPermDisplayNode = slicer.vtkMRMLModelDisplayNode()
    # modelPermDisplayNode.SetScene(slicer.mrmlScene)
    # slicer.mrmlScene.AddNode(modelPermDisplayNode)
    # modelNode.SetAndObserveDisplayNodeID(modelPermDisplayNode.GetID())
    # modelPermDisplayNode.SetRepresentation(modelPermDisplayNode.PointsRepresentation)
    # modelPermDisplayNode.SetPointSize(4)
    # modelPermDisplayNode.SetOpacity(1.0)
    # modelPermDisplayNode.SetColor(1,0,0)
    # modelPermDisplayNode.SetScalarVisibility(True)
    # modelPermDisplayNode.SetActiveScalarName("PointCloudColor")
    # modelPermDisplayNode.SetScalarRangeFlag(4)
    # # temp

  def calculateAndSetICPTransform(self):
    outputTrans = self.icpTransformSelector.currentNode()

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
    icp.GetLandmarkTransform().SetModeToSimilarity()
    icp.SetMeanDistanceModeToAbsoluteValue()
    icp.SetMaximumNumberOfIterations(int(self.icpIterationsSlider.value))
    icp.SetMaximumMeanDistance(self.icpMaxDistanceSlider.value)
    icp.SetMaximumNumberOfLandmarks(int(self.icpLandmarksSlider.value))
    icp.SetCheckMeanDistance(int(self.checkMeanDistanceCheckBox.isChecked()))
    icp.Update()

    slicer.mrmlScene.RemoveNode(hardenedFixedModelNode)
    slicer.mrmlScene.RemoveNode(hardenedMovingModelNode)

    outputMatrix = vtk.vtkMatrix4x4()
    icp.GetMatrix(outputMatrix)
    outputTrans.SetAndObserveMatrixTransformToParent(outputMatrix)

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