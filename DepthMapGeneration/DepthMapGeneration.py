# -*- coding: utf-8 -*-
from __main__ import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from PIL import Image, ImageDraw

import os
import time
import math
import cv2

import numpy as np

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
    self.FPSBox.value = 40
    controlLayout.addRow("Target Rate:", self.FPSBox)

    self.fovSliderWidget = ctk.ctkSliderWidget()
    self.fovSliderWidget.setDecimals(2)
    self.fovSliderWidget.minimum = 0.00
    self.fovSliderWidget.maximum = 720.00
    self.fovSliderWidget.singleStep = 0.01
    self.fovSliderWidget.value = 127.018
    controlLayout.addRow("Field of View:", self.fovSliderWidget)
    
    self.upAngleSliderWidget = ctk.ctkSliderWidget()
    self.upAngleSliderWidget.setDecimals(2)
    self.upAngleSliderWidget.minimum = 0.00
    self.upAngleSliderWidget.maximum = 720.00
    self.upAngleSliderWidget.singleStep = 1.00
    self.upAngleSliderWidget.value = 0.00
    controlLayout.addRow("Roll Angle:", self.upAngleSliderWidget)

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
    
    self.rgbImageSelector = slicer.qMRMLNodeComboBox()
    self.rgbImageSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.rgbImageSelector.selectNodeUponCreation = True
    self.rgbImageSelector.noneEnabled = True
    self.rgbImageSelector.addEnabled = True
    self.rgbImageSelector.showHidden = False
    self.rgbImageSelector.setMRMLScene( slicer.mrmlScene )
    controlLayout.addRow("RGB Node:", self.rgbImageSelector)
    
    self.undistortImageSelector = slicer.qMRMLNodeComboBox()
    self.undistortImageSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.undistortImageSelector.selectNodeUponCreation = True
    self.undistortImageSelector.noneEnabled = True
    self.undistortImageSelector.addEnabled = True
    self.undistortImageSelector.showHidden = False
    self.undistortImageSelector.setMRMLScene( slicer.mrmlScene )
    controlLayout.addRow("Undistortd Node:", self.undistortImageSelector)
    
    self.cameraRollTransformSelector = slicer.qMRMLNodeComboBox()
    self.cameraRollTransformSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.cameraRollTransformSelector.selectNodeUponCreation = False
    self.cameraRollTransformSelector.noneEnabled = False
    self.cameraRollTransformSelector.addEnabled = True
    self.cameraRollTransformSelector.removeEnabled = True
    self.cameraRollTransformSelector.setMRMLScene(slicer.mrmlScene)
    controlLayout.addRow("Camera Roll Transform: ", self.cameraRollTransformSelector)

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
    camera.SetClippingRange(0.1,100)

    forward = [cameraTransformMatrix.GetElement(0, 2), cameraTransformMatrix.GetElement(1, 2), cameraTransformMatrix.GetElement(2, 2)]
    position = [cameraTransformMatrix.GetElement(0, 3), cameraTransformMatrix.GetElement(1, 3), cameraTransformMatrix.GetElement(2, 3)]

    camera.SetFocalPoint(position[0] + forward[0], position[1] + forward[1], position[2] + forward[2])
    camera.SetPosition(position)
    #camera.SetViewUp(self.angle_to_unit_vector(self.upAngleSliderWidget.value))
    camera.SetViewUp([0.0,0.0,1.0])
    
    wcx = -2*(97.48221807028696 - (200)/2) / 200
    wcy =  2*(101.56844672451034 - (200)/2) / 200
    camera.SetWindowCenter(wcx, wcy)
    
    view_angle = vtk.vtkMath.DegreesFromRadians(2.0 * math.atan2( 200/2.0, self.fovSliderWidget.value ))
    camera.SetViewAngle(view_angle)

    self.setCameraRoll()

  def start(self):
    self.depthTimer.start(int(1000/int(self.FPSBox.value)))

  def stop(self):
    self.depthTimer.stop()
    
  def angle_to_unit_vector(self, angle):
    vector = np.array([0.0, 1.0, 0.0])
    #axis = np.array([0.0, 1.0, 0.0]) 
    axis = np.array([0.0, 0.0, 1.0]) 
    
    transform = vtk.vtkTransform()
    transform.RotateWXYZ(angle, axis)
    
    return transform.TransformVector(vector)
  
  def setCameraRoll(self):
    # cameraRollNode = self.cameraRollTransformSelector.currentNode()
    # cameraTransformNode = self.cameraTransformSelector.currentNode()
    upAngleVector = self.angle_to_unit_vector(self.upAngleSliderWidget.value) # Desired view up vector
    # cameraMatrix = vtk.vtkMatrix4x4()
    # cameraTransformNode.GetMatrixTransformToWorld(cameraMatrix)
    # currentUpVector = cameraMatrix.MultiplyPoint([0,0,1,0])
    # currentUpVector = [currentUpVector[0], currentUpVector[1], currentUpVector[2]]
    # betweenAngleRad = vtk.vtkMath.AngleBetweenVectors(currentUpVector, upAngleVector)
    # betweenAngleDeg = vtk.vtkMath.DegreesFromRadians(betweenAngleRad)
    
    # When replaying the data just make sure the SetViewUp is set to what it was to visually match 
    cameraNode = self.cameraSelector.currentNode()
    camera = cameraNode.GetCamera()
    camera.SetViewUp(upAngleVector)
    
    # if self.upAngleSliderWidget.value > 120:
      # betweenAngleRad += math.pi

    
    # cameraRollMatrix = vtk.vtkMatrix4x4()
    # cameraRollMatrix.SetElement(0,0,math.cos(-betweenAngleRad))
    # cameraRollMatrix.SetElement(0,1,-math.sin(-betweenAngleRad))
    # cameraRollMatrix.SetElement(1,0,math.sin(-betweenAngleRad))
    # cameraRollMatrix.SetElement(1,1,math.cos(-betweenAngleRad))
    
    # cameraRollNode.SetAndObserveMatrixTransformToParent(cameraRollMatrix)
  
  def undistortImage(self):
    self.setCameraRoll()
  
    rgbNode = self.rgbImageSelector.currentNode()
    vtk_image_data = rgbNode.GetImageData()
    
    # Convert vtkImageData to a numpy array (cv::Mat)
    # Get the scalar data (assuming it's RGB)
    img_scalar = vtk_image_data.GetPointData().GetScalars()

    # Get dimensions
    dims = vtk_image_data.GetDimensions()
    n_comp = img_scalar.GetNumberOfComponents()

    # Convert to numpy array
    temp = vtk.util.numpy_support.vtk_to_numpy(img_scalar)
    numpy_data = temp.reshape(dims[1], dims[0], n_comp)
    numpy_data = numpy_data.transpose(0, 1, 2)
    
    # Convert to OpenCV format (BGR)
    opencv_image = cv2.cvtColor(numpy_data, cv2.IMREAD_COLOR)

    # Undistort the image using OpenCV
    h, w = opencv_image.shape[:2]
    
    mtx = np.array([[127.01794355133495, 0.0, 97.48221807028696],
                              [0.0, 126.9569499131192, 101.56844672451034],
                              [0, 0, 1]])
    dist = np.array([[-0.10060237013732572, -0.15188797045217614, 0.0009562576409102434, -0.004886256805686288, 0.1019560097811537]])
    dst = cv2.undistort(opencv_image, mtx, dist, None)

    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(dst.shape[1], dst.shape[0], 1)
    vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
    
    #flattened_dst = dst.flatten()
    vtk_array = vtk.util.numpy_support.numpy_to_vtk(dst.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_array.SetNumberOfComponents(3)
    vtk_image.GetPointData().SetScalars(vtk_array)

    # # Copy pixel data from the OpenCV image to vtkImageData
    # for y in range(dst.shape[0]):
        # for x in range(dst.shape[1]):
            # pixel = dst[y, x]
            # vtk_image.SetScalarComponentFromFloat(x, y, 0, 0, pixel[0])
            # vtk_image.SetScalarComponentFromFloat(x, y, 0, 1, pixel[1])
            # vtk_image.SetScalarComponentFromFloat(x, y, 0, 2, pixel[2])

    newRgbNode = self.undistortImageSelector.currentNode()
    newRgbNode.SetAndObserveImageData(vtk_image)
    #newRgbNode.SetIJKToRASDirections(1,0,0,0,1,0,0,0,1)
  
  def screen_to_eye_depth_perspective(self, d, zNear, zFar):
    depth = d * 2.0 - 1.0
    return  (2.0 * zNear * zFar) / (zFar + zNear - depth * (zFar - zNear))

  def generateDepth(self):
    cameraNode = self.cameraSelector.currentNode()
    
    # Fix roll
    camera = cameraNode.GetCamera()
    #camera.SetViewUp(self.angle_to_unit_vector(self.upAngleSliderWidget.value))
    self.setCameraRoll()
    
    # Undistort
    self.undistortImage()
    
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
      flip_filter.SetFilteredAxis(0)  # Flip along y-axis
      flip_filter.Update()
        
      resize = vtk.vtkImageResize()
      resize.SetResizeMethodToOutputDimensions()
      resize.SetInputData(flip_filter.GetOutput())
      resize.SetOutputDimensions(200, 200, 1)
      resize.Update()

      # Convert to Linear
      vtkImage = resize.GetOutput()
      vtk_data_array = vtkImage.GetPointData().GetScalars()
      numpy_array = vtk.util.numpy_support.vtk_to_numpy(vtk_data_array)
      zNear = camera.GetClippingRange()[0]
      zFar = camera.GetClippingRange()[1]
      numpy_array = self.screen_to_eye_depth_perspective(numpy_array, zNear, zFar)
      vtk_data_array = vtk.util.numpy_support.numpy_to_vtk(numpy_array)
      vtkImage.GetPointData().SetScalars(vtk_data_array)

      change = vtk.vtkImageChangeInformation()
      change.SetInputConnection(resize.GetOutputPort())
      change.SetOutputSpacing(cropped_image.GetSpacing())
      change.Update()
      
      outputNode.SetAndObserveImageData(change.GetOutput())
      outputNode.SetIJKToRASDirections(1,0,0,0,1,0,0,0,1)

class DepthMapGenerationLogic:
  def __init__(self):
    pass

