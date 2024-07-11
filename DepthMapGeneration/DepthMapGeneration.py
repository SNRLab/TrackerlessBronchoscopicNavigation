# -*- coding: utf-8 -*-
from __main__ import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import math
from PIL import Image, ImageDraw

import os
import math
import time

class DepthMapGeneration(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    parent.title = "Depth Map Generation"
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
    

class DepthMapGenerationWidget(ScriptedLoadableModuleWidget):
  def __init__(self, parent = None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    self.lastCommandId = 0
    self.timeoutCounter = 0
  
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

    self.fovSliderWidget = ctk.ctkSliderWidget()
    self.fovSliderWidget.setDecimals(2)
    self.fovSliderWidget.minimum = 0.00
    self.fovSliderWidget.maximum = 360.00
    self.fovSliderWidget.singleStep = 0.01
    self.fovSliderWidget.value = 45.00
    controlLayout.addRow("Field of View:", self.fovSliderWidget)

    self.setCameraButton = qt.QPushButton("Set Camera")
    controlLayout.addRow(self.setCameraButton)
    self.setCameraButton.connect('clicked(bool)', self.setCamera)    
    
    self.depthTimer = qt.QTimer()
    self.depthTimer.timeout.connect(self.generateDepth)

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

    self.layout.addStretch(1)

  def setCamera(self):
    cameraNode = self.cameraSelector.currentNode()
    cameraTransformNode = self.cameraTransformSelector.currentNode()
    cameraTransformMatrix = vtk.vtkMatrix4x4()
    cameraTransformNode.GetMatrixTransformToWorld(cameraTransformMatrix)
    camera = cameraNode.GetCamera()

    forward = [cameraTransformMatrix.GetElement(0, 2), cameraTransformMatrix.GetElement(1, 2), cameraTransformMatrix.GetElement(2, 2)]
    position = [cameraTransformMatrix.GetElement(0, 3), cameraTransformMatrix.GetElement(1, 3), cameraTransformMatrix.GetElement(2, 3)]

    camera.SetFocalPoint(position[0] + forward[0], position[1] + forward[1], position[2] + forward[2])
    camera.SetPosition(position)
    camera.SetViewUp(0, 0, 1)
    camera.SetViewAngle(self.fovSliderWidget.value)

  def start(self):
    self.depthTimer.start(int(1000/int(self.FPSBox.value)))

  def stop(self):
    self.depthTimer.stop()

  def generateDepth(self):
    cameraNode = self.cameraSelector.currentNode()
    outputNode = self.outputImageSelector.currentNode()
    renderWindow = None
    layoutManager = slicer.app.layoutManager()
    for viewIndex in range(layoutManager.threeDViewCount):
      threeDWidget = layoutManager.threeDWidget(viewIndex)
      viewCameraNode = threeDWidget.threeDView().cameraNode()
      if viewCameraNode == cameraNode:
        renderWindow = threeDWidget.threeDView().renderWindow()
        break
    
    if renderWindow:
      windowToImageFilter = vtk.vtkWindowToImageFilter()
      windowToImageFilter.SetInput(renderWindow)
      windowToImageFilter.SetInputBufferTypeToZBuffer()
      windowToImageFilter.Update()
      depth_map = windowToImageFilter.GetOutput()

      dims = depth_map.GetDimensions()
      center_square_dim = dims[1]

      # Calculate the extents of the center square
      x_min = dims[0] // 2 - center_square_dim // 2
      x_max = dims[0] // 2 + center_square_dim // 2
      y_min = dims[1] // 2 - center_square_dim // 2
      y_max = dims[1] // 2 + center_square_dim // 2

      extract_voi = vtk.vtkExtractVOI()
      extract_voi.SetInputData(depth_map)
      extract_voi.SetVOI(x_min, x_max, y_min, y_max, 0, 0)
      extract_voi.Update()

      # Get the output of vtkExtractVOI
      cropped_image = extract_voi.GetOutput()

      flip_filter = vtk.vtkImageFlip()
      flip_filter.SetInputData(cropped_image)
      # Set the flip axis (0 for x-axis, 1 for y-axis, 2 for z-axis)
      flip_filter.SetFilteredAxis(1)  # Flip along y-axis
      flip_filter.Update()
      
      # flip_filter2 = vtk.vtkImageFlip()
      # flip_filter2.SetInputData(flip_filter.GetOutput())
      # # Set the flip axis (0 for x-axis, 1 for y-axis, 2 for z-axis)
      # flip_filter2.SetFilteredAxis(0)  # Flip along y-axis
      # flip_filter2.Update()      
        
      resize = vtk.vtkImageResize()
      resize.SetResizeMethodToOutputDimensions();
      resize.SetInputData(flip_filter.GetOutput())
      resize.SetOutputDimensions(200, 200, 1)
      resize.Update()
      
      change = vtk.vtkImageChangeInformation()
      change.SetInputConnection(resize.GetOutputPort())
      change.SetOutputSpacing(cropped_image.GetSpacing())
      change.Update()

      outputNode.SetAndObserveImageData(change.GetOutput())
      outputNode.SetIJKToRASDirections(-1,0,0,0,-1,0,0,0,1)

      # # If you want to save the depth map as an image, you can use vtkPNGWriter
      # writer = vtk.vtkPNGWriter()
      # writer.SetFileName("depth_map.png")
      # writer.SetInputData(depth_map)
      # writer.Write()

      # # Print a success message
      # print("The depth map was successfully saved as depth_map.png")

class DepthMapGenerationLogic:
  def __init__(self):
    pass

