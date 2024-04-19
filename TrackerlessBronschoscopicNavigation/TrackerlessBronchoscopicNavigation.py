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

class TrackerlessBronchoscopicNavigation(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Bakse/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Trackerless Bronchoscopic Navigation"
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

class TrackerlessBronchoscopicNavigationWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    self.stepCount = 0
    self.stepTimer = qt.QTimer()
    self.stepTimer.timeout.connect(self.onStepImage)

  def cleanup(self):
    self.stepTimer.stop()
    self.onResetStepCount()

  def onReload(self,moduleName="TrackerlessBronchoscopicNavigation"):
    self.stepTimer.stop()
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

    # self.inputImageSelector = slicer.qMRMLNodeComboBox()
    # self.inputImageSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    # self.inputImageSelector.selectNodeUponCreation = False
    # self.inputImageSelector.noneEnabled = True
    # self.inputImageSelector.addEnabled = True
    # self.inputImageSelector.removeEnabled = True
    # self.inputImageSelector.setMRMLScene(slicer.mrmlScene)
    # IOLayout.addRow("Camera Image: ", self.inputImageSelector)

    self.inputTransformSelector = slicer.qMRMLNodeComboBox()
    self.inputTransformSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.inputTransformSelector.selectNodeUponCreation = False
    self.inputTransformSelector.noneEnabled = False
    self.inputTransformSelector.addEnabled = True
    self.inputTransformSelector.removeEnabled = True
    self.inputTransformSelector.setMRMLScene(slicer.mrmlScene)
    IOLayout.addRow("Camera Transform: ", self.inputTransformSelector)    

    # Step mode collapsible button
    self.stepModeCollapsibleButton = ctk.ctkCollapsibleButton()
    self.stepModeCollapsibleButton.text = "Step Mode"
    self.stepModeCollapsibleButton.collapsed = False
    self.layout.addWidget(self.stepModeCollapsibleButton)
    stepModeLayout = qt.QFormLayout(self.stepModeCollapsibleButton)

    self.predPathBox = qt.QLineEdit("D:/MeghaData/data/dataset_14")
    self.predPathBox.setReadOnly(True)
    self.predPathButton = qt.QPushButton("...")
    self.predPathButton.clicked.connect(self.select_directory_pred)
    predPathBoxLayout = qt.QHBoxLayout()
    predPathBoxLayout.addWidget(self.predPathBox)
    predPathBoxLayout.addWidget(self.predPathButton)
    stepModeLayout.addRow(predPathBoxLayout)

    self.loadPredButton = qt.QPushButton("Load Predictions")
    stepModeLayout.addWidget(self.loadPredButton)
    self.loadPredButton.connect('clicked()', self.onLoadPred)

    self.gtCheckBox = qt.QCheckBox("Use Ground Truth Data")
    self.gtCheckBox.setChecked(False)
    stepModeLayout.addWidget(self.gtCheckBox)

    self.gtPathBox = qt.QLineEdit("D:/MeghaData/Data/dataset_14/keyframe_1")
    self.gtPathBox.setReadOnly(True)
    self.gtPathButton = qt.QPushButton("...")
    self.gtPathButton.clicked.connect(self.select_directory_gt)
    gtPathBoxLayout = qt.QHBoxLayout()
    gtPathBoxLayout.addWidget(self.gtPathBox)
    gtPathBoxLayout.addWidget(self.gtPathButton)
    stepModeLayout.addRow(gtPathBoxLayout)

    self.stepButton = qt.QPushButton("Step Image")
    stepModeLayout.addWidget(self.stepButton)
    self.stepButton.connect('clicked()', self.onStepImage)

    self.resetStepButton = qt.QPushButton("Reset Step Count")
    stepModeLayout.addWidget(self.resetStepButton)
    self.resetStepButton.connect('clicked()', self.onResetStepCount)

    self.stepLabel = qt.QLabel("0")
    stepModeLayout.addWidget(self.stepLabel)

    self.stepTimerButton = qt.QPushButton("Step Timer")
    self.stepTimerButton.setCheckable(True)
    stepModeLayout.addWidget(self.stepTimerButton)
    self.stepTimerButton.connect('clicked()', self.onStepTimer)

    self.stepFPSBox = qt.QSpinBox()
    self.stepFPSBox.setSingleStep(1)
    self.stepFPSBox.setMaximum(144)
    self.stepFPSBox.setMinimum(1)
    self.stepFPSBox.setSuffix(" FPS")
    self.stepFPSBox.value = 60
    stepModeLayout.addWidget(self.stepFPSBox)

    self.scaleSliderWidget = ctk.ctkSliderWidget()
    self.scaleSliderWidget.setDecimals(2)
    self.scaleSliderWidget.minimum = 0.00
    self.scaleSliderWidget.maximum = 10000.00
    self.scaleSliderWidget.singleStep = 0.01
    self.scaleSliderWidget.value = 200.00
    stepModeLayout.addRow("Scale Factor:", self.scaleSliderWidget)

    # Add vertical spacer
    self.layout.addStretch(1)

  def select_directory_pred(self):
    directory = qt.QFileDialog.getExistingDirectory(self.parent, "Select Directory")
    if directory:
      self.predPathBox.setText(directory)

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
    self.stepCount = 0
    self.stepLabel.setText('0')

    layoutManager = slicer.app.layoutManager()
    red = layoutManager.sliceWidget('Red')
    redLogic = red.sliceLogic()
    redLogic.SetSliceOffset(0)

    green = layoutManager.sliceWidget('Green')
    greenLogic = green.sliceLogic()
    greenLogic.SetSliceOffset(0)
    
    resultMatrix = vtk.vtkMatrix4x4()
    self.inputTransformSelector.currentNode().SetAndObserveMatrixTransformToParent(resultMatrix)
  
  def onLoadPred(self):
    self.translations_pred = np.load(f'{self.predPathBox.text}/translation.npz')
    self.axis_angle_pred = np.load(f'{self.predPathBox.text}/axisangle.npz')
    self.pose_pred = np.load(f'{self.predPathBox.text}/pose_prediction.npz')


  def onStepImage(self):
    import re
    import json
    self.stepCount = self.stepCount + 1
    stepCountString = str(self.stepCount)
    self.stepLabel.setText(stepCountString)
    
    if self.gtCheckBox.isChecked():
      prefix = "frame_data"
      # imageFilename = ""
      frameDataFilename = ""
      for filename in os.listdir(self.gtPathBox.text):
        # if filename.lstrip('0').split('.')[0] == stepCountString:
        #   imageFilename = filename
        
        if filename.startswith(prefix):
          stripped_filename = re.sub('^' + prefix, '', filename).lstrip('0').split('.')[0]
          if stripped_filename == stepCountString:
            frameDataFilename = filename
          
      # # Display image by moving slice offset
      #self.inputImageSelector.currentNode()
      layoutManager = slicer.app.layoutManager()
      red = layoutManager.sliceWidget('Red')
      redLogic = red.sliceLogic()
      redLogic.SetSliceOffset(self.stepCount - 1)

      green = layoutManager.sliceWidget('Green')
      greenLogic = green.sliceLogic()
      greenLogic.SetSliceOffset(self.stepCount - 1)

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
    else:
      # Display image by moving slice offset
      layoutManager = slicer.app.layoutManager()
      red = layoutManager.sliceWidget('Red')
      redLogic = red.sliceLogic()
      redLogic.SetSliceOffset(self.stepCount - 1)

      green = layoutManager.sliceWidget('Green')
      greenLogic = green.sliceLogic()
      greenLogic.SetSliceOffset(self.stepCount - 1)

      scale = self.scaleSliderWidget.value

      previousMatrix = vtk.vtkMatrix4x4()
      self.inputTransformSelector.currentNode().GetMatrixTransformToParent(previousMatrix)

      previousRotationMatrix = vtk.vtkMatrix4x4()
      for i in range(3):
        for j in range(3):
          previousRotationMatrix.SetElement(i, j, previousMatrix.GetElement(i, j))

      # Just use the o
      pred_poses = []
      pred_poses.append(layers.transformation_from_parameters(torch.from_numpy(self.axis_angle_pred['arr_0'][self.stepCount-1:self.stepCount, 0]), torch.from_numpy(self.translations_pred['arr_0'][self.stepCount-1:self.stepCount, 0]) * scale).cpu().numpy())
      pred_poses = np.concatenate(pred_poses)
      dump_our = np.array(self.dump(self.vtk_to_numpy_matrix(previousMatrix), pred_poses))

      #print(dump_our[1])

      self.inputTransformSelector.currentNode().SetAndObserveMatrixTransformToParent(self.numpy_to_vtk_matrix(dump_our[1]))

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
