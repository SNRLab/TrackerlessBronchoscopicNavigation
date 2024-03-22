import os
from pyexpat import model
import unittest
# from matplotlib.pyplot import get
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from vtk.util import numpy_support
import numpy as np
from sys import platform
from PIL import Image

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

    self.predPathBox = qt.QLineEdit("D:/MeghaData/predictions")
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

    self.gtPathBox = qt.QLineEdit("D:/MeghaData/predictions/dataset_14/keyframe_1")
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
    self.stepFPSBox.value = 10
    stepModeLayout.addWidget(self.stepFPSBox)

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
  
  def onLoadPred(self):
    self.translations_pred = np.load(f'{self.predPathBox.text}/translation_prediction.npz')
    self.axis_angle_pred = np.load(f'{self.predPathBox.text}/axis_angle_prediction.npz')

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

      newMatrix = self.axis_angle_to_matrix_with_translation(self.axis_angle_pred['arr_0'][self.stepCount-1][0][0], 0, self.translations_pred['arr_0'][self.stepCount-1][0][0])
      oldMatrix = vtk.vtkMatrix4x4()
      self.inputTransformSelector.currentNode().GetMatrixTransformToParent(oldMatrix)
      resultMatrix = vtk.vtkMatrix4x4()
      vtk.vtkMatrix4x4.Multiply4x4(oldMatrix, newMatrix, resultMatrix)
      self.inputTransformSelector.currentNode().SetAndObserveMatrixTransformToParent(resultMatrix)

  def axis_angle_to_matrix_with_translation(self, a, theta, t):
    # Ensure the axis is a unit vector
    a = a / np.linalg.norm(a)

    a1, a2, a3 = a
    c = np.cos(theta)
    s = np.sin(theta)
    t1, t2, t3 = t
    one_c = 1 - c

    m00 = c + a1*a1*one_c
    m11 = c + a2*a2*one_c
    m22 = c + a3*a3*one_c

    tmp1 = a1*a2*one_c
    tmp2 = a3*s
    m10 = tmp1 + tmp2
    m01 = tmp1 - tmp2

    tmp1 = a1*a3*one_c
    tmp2 = a2*s
    m20 = tmp1 - tmp2
    m02 = tmp1 + tmp2

    tmp1 = a2*a3*one_c
    tmp2 = a1*s
    m21 = tmp1 + tmp2
    m12 = tmp1 - tmp2

    matrix = vtk.vtkMatrix4x4()
    matrix.SetElement(0, 0, m00); matrix.SetElement(0, 1, m01); matrix.SetElement(0, 2, m02); matrix.SetElement(0, 3, t1)
    matrix.SetElement(1, 0, m10); matrix.SetElement(1, 1, m11); matrix.SetElement(1, 2, m12); matrix.SetElement(0, 3, t2)
    matrix.SetElement(2, 0, m20); matrix.SetElement(2, 1, m21); matrix.SetElement(2, 2, m22); matrix.SetElement(0, 3, t3)
    matrix.SetElement(3, 0, 0); matrix.SetElement(3, 1, 0); matrix.SetElement(3, 2, 0); matrix.SetElement(0, 3, 1)

    return matrix