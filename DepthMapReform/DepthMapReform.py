# -*- coding: utf-8 -*-
from __main__ import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import math
from PIL import Image, ImageDraw

import os
import math
import time

class DepthMapReform(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    parent.title = "Depth Map Reform"
    parent.categories = ["Utilities"]
    parent.contributors = ["Franklin King"]
    parent.helpText = """
    Add help text
    """
    parent.acknowledgementText = """
""" 
    # module build directory is not the current directory when running the python script, hence why the usual method of finding resources didn't work ASAP
    self.parent = parent
    
    # Set module icon from Resources/Icons/<ModuleName>.png
    moduleDir = os.path.dirname(self.parent.path)
    for iconExtension in ['.svg', '.png']:
      iconPath = os.path.join(moduleDir, 'Resources/Icons', self.__class__.__name__ + iconExtension)
      if os.path.isfile(iconPath):
        parent.icon = qt.QIcon(iconPath)
        break
    

class DepthMapReformWidget(ScriptedLoadableModuleWidget):
  def __init__(self, parent = None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    self.logIndex = 0
    self.lines = None
  
  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
  
    LayoutButton = ctk.ctkCollapsibleButton()
    LayoutButton.text = "Control"
    self.layout.addWidget(LayoutButton)
    
    controlLayout = qt.QFormLayout(LayoutButton)

    self.startButton = qt.QPushButton("Start")
    controlLayout.addRow(self.startButton)
    self.startButton.connect('clicked(bool)', self.start)    

    self.stopButton = qt.QPushButton("Stop")
    controlLayout.addRow(self.stopButton)
    self.stopButton.connect('clicked(bool)', self.stop)
    
    self.FPSBox = qt.QSpinBox()
    self.FPSBox.setSingleStep(1)
    self.FPSBox.setMaximum(144)
    self.FPSBox.setMinimum(1)
    self.FPSBox.setSuffix(" FPS")
    self.FPSBox.value = 20
    controlLayout.addRow("Target Rate:", self.FPSBox)
    
    self.depthTimer = qt.QTimer()
    self.depthTimer.timeout.connect(self.depth)   

    self.cameraSelector = slicer.qMRMLNodeComboBox()
    self.cameraSelector.nodeTypes = ["vtkMRMLCameraNode"]
    self.cameraSelector.selectNodeUponCreation = True
    self.cameraSelector.noneEnabled = True
    self.cameraSelector.addEnabled = True
    self.cameraSelector.showHidden = False
    self.cameraSelector.setMRMLScene( slicer.mrmlScene )
    controlLayout.addRow("Camera Node:", self.cameraSelector)

    self.cameraTransformSelector = slicer.qMRMLNodeComboBox()
    self.cameraTransformSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.cameraTransformSelector.selectNodeUponCreation = False
    self.cameraTransformSelector.noneEnabled = False
    self.cameraTransformSelector.addEnabled = True
    self.cameraTransformSelector.removeEnabled = True
    self.cameraTransformSelector.setMRMLScene(slicer.mrmlScene)
    controlLayout.addRow("Camera Transform: ", self.cameraTransformSelector)

    self.outputImageSelector = slicer.qMRMLNodeComboBox()
    self.outputImageSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.outputImageSelector.selectNodeUponCreation = False
    self.outputImageSelector.noneEnabled = False
    self.outputImageSelector.addEnabled = True
    self.outputImageSelector.removeEnabled = True
    self.outputImageSelector.setMRMLScene(slicer.mrmlScene)
    controlLayout.addRow("Depth Map: ", self.outputImageSelector)

    self.logPathBox = qt.QLineEdit()
    self.logBrowseButton = qt.QPushButton("...")
    self.logBrowseButton.clicked.connect(self.select_directory)
    pathBoxLayout = qt.QHBoxLayout()
    pathBoxLayout.addWidget(self.logPathBox)
    pathBoxLayout.addWidget(self.logBrowseButton)
    controlLayout.addRow(pathBoxLayout)

    self.layout.addStretch(1)

  def select_directory(self):
    directory = qt.QFileDialog.getExistingDirectory(self.parent, "Select Directory")
    if directory:
      self.logPathBox.setText(directory)

  def start(self):
    logFile = open(f'{self.logPathBox.text}/log.txt', 'r')
    self.lines = logFile.readlines()
    self.depthTimer.start(int(1000/int(self.FPSBox.value)))
     
  def depth(self):
    cameraTransformNode = self.cameraTransformSelector.currentNode()
    name = self.lines[self.logIndex]
    timestamp = name.split('_')[-1].rstrip()
    matrixStr = [self.lines[self.logIndex+1].split(' '), self.lines[self.logIndex+2].split(' '), self.lines[self.logIndex+3].split(' '), self.lines[self.logIndex+4].split(' ')]
    matrix = vtk.vtkMatrix4x4()
    matrix.SetElement(0,0,float(matrixStr[0][0])); matrix.SetElement(0,1,float(matrixStr[0][1])); matrix.SetElement(0,2,float(matrixStr[0][2])); matrix.SetElement(0,3,float(matrixStr[0][3]))
    matrix.SetElement(1,0,float(matrixStr[1][0])); matrix.SetElement(1,1,float(matrixStr[1][1])); matrix.SetElement(1,2,float(matrixStr[1][2])); matrix.SetElement(1,3,float(matrixStr[1][3]))
    matrix.SetElement(2,0,float(matrixStr[2][0])); matrix.SetElement(2,1,float(matrixStr[2][1])); matrix.SetElement(2,2,float(matrixStr[2][2])); matrix.SetElement(2,3,float(matrixStr[2][3]))
    matrix.SetElement(3,0,float(matrixStr[3][0])); matrix.SetElement(3,1,float(matrixStr[3][1])); matrix.SetElement(3,2,float(matrixStr[3][2])); matrix.SetElement(3,3,float(matrixStr[3][3]))
    cameraTransformNode.SetMatrixTransformToParent(matrix)

    outputName = f'{self.logPathBox.text}/depth_{timestamp}.tiff'

    depthImageData = self.outputImageSelector.currentNode().GetImageData()

    flip_filter = vtk.vtkImageFlip()
    flip_filter.SetInputData(depthImageData)
    # Set the flip axis (0 for x-axis, 1 for y-axis, 2 for z-axis)
    flip_filter.SetFilteredAxis(0)  # Flip along y-axis
    flip_filter.Update()

    depthWriter = vtk.vtkTIFFWriter()
    depthWriter.SetCompressionToNoCompression()
    depthWriter.SetInputData(flip_filter.GetOutput())
    depthWriter.SetFileName(outputName)
    depthWriter.Write()

    self.logIndex += 6

  def stop(self):
    self.logIndex = 0
    self.depthTimer.stop()

class DepthMapReformLogic:
  def __init__(self):
    pass

