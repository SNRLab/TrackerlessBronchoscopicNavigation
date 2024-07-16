# -*- coding: utf-8 -*-
from __main__ import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import math
from PIL import Image, ImageDraw

import os
import math
import time

class EndoscopeRecord(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    parent.title = "Endoscope Record"
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
    

class EndoscopeRecordWidget(ScriptedLoadableModuleWidget):
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
    
    self.recordTimer = qt.QTimer()
    self.recordTimer.timeout.connect(self.record)   

    self.rgbSelector = slicer.qMRMLNodeComboBox()
    self.rgbSelector.nodeTypes = ["vtkMRMLVectorVolumeNode"]
    self.rgbSelector.selectNodeUponCreation = False
    self.rgbSelector.noneEnabled = True
    self.rgbSelector.addEnabled = False
    self.rgbSelector.showHidden = False
    self.rgbSelector.setMRMLScene( slicer.mrmlScene )
    controlLayout.addRow("RGB Image:", self.rgbSelector)

    self.grayscaleSelector = slicer.qMRMLNodeComboBox()
    self.grayscaleSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.grayscaleSelector.selectNodeUponCreation = False
    self.grayscaleSelector.noneEnabled = True
    self.grayscaleSelector.addEnabled = False
    self.grayscaleSelector.showHidden = False
    self.grayscaleSelector.setMRMLScene( slicer.mrmlScene )
    controlLayout.addRow("Grayscale Image:", self.grayscaleSelector)
    
    self.depthSelector = slicer.qMRMLNodeComboBox()
    self.depthSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.depthSelector.selectNodeUponCreation = False
    self.depthSelector.noneEnabled = True
    self.depthSelector.addEnabled = False
    self.depthSelector.showHidden = False
    self.depthSelector.setMRMLScene( slicer.mrmlScene )
    controlLayout.addRow("Depth Image:", self.depthSelector)    

    self.transformSelector = slicer.qMRMLNodeComboBox()
    self.transformSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.transformSelector.selectNodeUponCreation = True
    self.transformSelector.noneEnabled = True
    self.transformSelector.addEnabled = True
    self.transformSelector.showHidden = False
    self.transformSelector.setMRMLScene( slicer.mrmlScene )
    controlLayout.addRow("Catheter Transform 5DoF:", self.transformSelector)
    
    self.transformSelector2 = slicer.qMRMLNodeComboBox()
    self.transformSelector2.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.transformSelector2.selectNodeUponCreation = True
    self.transformSelector2.noneEnabled = True
    self.transformSelector2.addEnabled = True
    self.transformSelector2.showHidden = False
    self.transformSelector2.setMRMLScene( slicer.mrmlScene )
    controlLayout.addRow("Target 1 Transform 5DoF:", self.transformSelector2)
    
    self.transformSelector3 = slicer.qMRMLNodeComboBox()
    self.transformSelector3.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.transformSelector3.selectNodeUponCreation = True
    self.transformSelector3.noneEnabled = True
    self.transformSelector3.addEnabled = True
    self.transformSelector3.showHidden = False
    self.transformSelector3.setMRMLScene( slicer.mrmlScene )
    controlLayout.addRow("Target 2 Transform 5DoF:", self.transformSelector3)
    
    self.transformSelector4 = slicer.qMRMLNodeComboBox()
    self.transformSelector4.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.transformSelector4.selectNodeUponCreation = True
    self.transformSelector4.noneEnabled = True
    self.transformSelector4.addEnabled = True
    self.transformSelector4.showHidden = False
    self.transformSelector4.setMRMLScene( slicer.mrmlScene )
    controlLayout.addRow("Target 3 Transform 5DoF:", self.transformSelector4)
    
    self.transformSelector5 = slicer.qMRMLNodeComboBox()
    self.transformSelector5.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.transformSelector5.selectNodeUponCreation = True
    self.transformSelector5.noneEnabled = True
    self.transformSelector5.addEnabled = True
    self.transformSelector5.showHidden = False
    self.transformSelector5.setMRMLScene( slicer.mrmlScene )
    controlLayout.addRow("Target 4 Transform 5DoF:", self.transformSelector5)

    self.inputsFiducialSelector = slicer.qMRMLNodeComboBox()
    self.inputsFiducialSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
    self.inputsFiducialSelector.selectNodeUponCreation = False
    self.inputsFiducialSelector.noneEnabled = True
    self.inputsFiducialSelector.addEnabled = False
    self.inputsFiducialSelector.showHidden = False
    self.inputsFiducialSelector.setMRMLScene( slicer.mrmlScene )
    controlLayout.addRow("Inputs:", self.inputsFiducialSelector)   

    self.imagePathBox = qt.QLineEdit()
    self.imageBrowseButton = qt.QPushButton("...")
    self.imageBrowseButton.clicked.connect(self.select_directory)
    pathBoxLayout = qt.QHBoxLayout()
    pathBoxLayout.addWidget(self.imagePathBox)
    pathBoxLayout.addWidget(self.imageBrowseButton)
    controlLayout.addRow(pathBoxLayout)

    self.layout.addStretch(1)

  def select_directory(self):
    directory = qt.QFileDialog.getExistingDirectory(self.parent, "Select Directory")
    if directory:
      self.imagePathBox.setText(directory)
      
  def select_directory2(self):
    directory = qt.QFileDialog.getExistingDirectory(self.parent, "Select Directory")
    if directory:
      self.imagePathBox2.setText(directory)      

  def start(self):
    self.recordTimer.start(int(1000/int(self.FPSBox.value)))

  def stop(self):
    self.recordTimer.stop()

  def generateInputsImage(self):
    inputsFiducial = self.inputsFiducialSelector.currentNode()
    inputs = [0,0,0]
    inputsFiducial.GetNthFiducialPosition(0, inputs)
    
    # image_size = (200, 200)
    # center = (image_size[0]//2, image_size[1]//2)
    # image = Image.new('L', image_size, 'gray')
    # draw = ImageDraw.Draw(image)

    # circle_diameter = 40
    # circle_brightness = int((inputs[2] + 1) * 127.5)
    # circle_coords = [(image_size[0]/2 - circle_diameter/2, image_size[1]/2 - circle_diameter/2), (image_size[0]/2 + circle_diameter/2, image_size[1]/2 + circle_diameter/2)]
    # draw.ellipse(circle_coords, fill=circle_brightness)

    # draw.line([center, (inputs[0]*image_size[0]+(image_size[0]//2),inputs[1]*image_size[1]+(image_size[1]//2))], fill='white', width = 8)
    
    #return image

    image_size = (200, 200)
    center = (image_size[0]//2, image_size[1]//2)

    linear_image = Image.new('L', image_size, 'gray')
    linear_draw = ImageDraw.Draw(linear_image)

    circle_diameter = 900
    circle_brightness = int((inputs[2] + 1) * 127.5)
    circle_coords = [(image_size[0]/2 - circle_diameter/2, image_size[1]/2 - circle_diameter/2), (image_size[0]/2 + circle_diameter/2, image_size[1]/2 + circle_diameter/2)]
    linear_draw.ellipse(circle_coords, fill=circle_brightness)

    direction_image = Image.new('L', image_size, 'black')
    direction_draw = ImageDraw.Draw(direction_image)

    direction_draw.line([center, (inputs[0]*image_size[0]+(image_size[0]//2),inputs[1]*image_size[1]+(image_size[1]//2))], fill='white', width = 12)

    return direction_image, linear_image

  def record(self):
    if self.imagePathBox.text:
      timestamp_millis = int(time.time() * 1000)
      filename = fr'{self.imagePathBox.text}/rgb_{timestamp_millis}'
      grayFilename = fr'{self.imagePathBox.text}/gray_{timestamp_millis}'
      depthFilename = fr'{self.imagePathBox.text}/depth_{timestamp_millis}'
      #inputsFilename = fr'{self.imagePathBox.text}/input_{timestamp_millis}'
      inputsDirectionFilename = fr'{self.imagePathBox.text}/inputDirection_{timestamp_millis}'
      inputsLinearFilename = fr'{self.imagePathBox.text}/inputLinear_{timestamp_millis}'
      rgbImageData = None
      grayscaleImageData = None
      depthImageData = None
      if self.rgbSelector.currentNode():
        rgbImageData = self.rgbSelector.currentNode().GetImageData()
        rgbWriter = vtk.vtkPNGWriter()
        rgbWriter.SetInputData(rgbImageData)
        rgbWriter.SetFileName(f'{filename}.png')
        rgbWriter.Write()
      if self.grayscaleSelector.currentNode():
        grayscaleImageData = self.grayscaleSelector.currentNode().GetImageData()
        table = vtk.vtkScalarsToColors()
        table.SetRange(0.0, 1.0)  # Set the range of your data values
        convert = vtk.vtkImageMapToColors()
        convert.SetLookupTable(table)
        convert.SetOutputFormatToRGB()
        convert.SetInputData(grayscaleImageData)
        convert.Update()
        grayscaleWriter = vtk.vtkPNGWriter()
        grayscaleWriter.SetInputData(convert.GetOutput())
        grayscaleWriter.SetFileName(f'{grayFilename}.png')
        grayscaleWriter.Write()
      if self.depthSelector.currentNode():
        depthImageData = self.depthSelector.currentNode().GetImageData()
        table = vtk.vtkScalarsToColors()
        table.SetRange(0.99, 1.0)  # Set the range of your data values
        convert = vtk.vtkImageMapToColors()
        convert.SetLookupTable(table)
        convert.SetOutputFormatToRGB()
        convert.SetInputData(depthImageData)
        convert.Update()
        depthWriter = vtk.vtkPNGWriter()
        depthWriter.SetInputData(convert.GetOutput())
        depthWriter.SetFileName(f'{depthFilename}.png')
        depthWriter.Write()
      if self.inputsFiducialSelector.currentNode():
        # inputsImageData = self.generateInputsImage()
        # inputsImageData.save(f'{inputsFilename}.png')
        inputsDirectionImageData, inputsLinearImageData = self.generateInputsImage()
        inputsDirectionImageData.save(f'{inputsDirectionFilename}.png')
        inputsLinearImageData.save(f'{inputsLinearFilename}.png')
      
      if self.transformSelector.currentNode():
        with open(f'{self.imagePathBox.text}/log.txt', 'a+') as file:
          matrix = vtk.vtkMatrix4x4()
          self.transformSelector.currentNode().GetMatrixTransformToWorld(matrix)
          matrixString = f'\n{matrix.GetElement(0,0)} {matrix.GetElement(0,1)} {matrix.GetElement(0,2)} {matrix.GetElement(0,3)}\n{matrix.GetElement(1,0)} {matrix.GetElement(1,1)} {matrix.GetElement(1,2)} {matrix.GetElement(1,3)}\n{matrix.GetElement(2,0)} {matrix.GetElement(2,1)} {matrix.GetElement(2,2)} {matrix.GetElement(2,3)}\n{matrix.GetElement(3,0)} {matrix.GetElement(3,1)} {matrix.GetElement(3,2)} {matrix.GetElement(3,3)}\n\n'
          file.write(f'5dof_catheter_rgb_{timestamp_millis}')
          file.write(matrixString)
      if self.transformSelector2.currentNode(): 
        with open(f'{self.imagePathBox.text}/log.txt', 'a+') as file:
          matrix2 = vtk.vtkMatrix4x4()
          self.transformSelector2.currentNode().GetMatrixTransformToWorld(matrix2)
          matrixString2 = f'\n{matrix2.GetElement(0,0)} {matrix2.GetElement(0,1)} {matrix2.GetElement(0,2)} {matrix2.GetElement(0,3)}\n{matrix2.GetElement(1,0)} {matrix2.GetElement(1,1)} {matrix2.GetElement(1,2)} {matrix2.GetElement(1,3)}\n{matrix2.GetElement(2,0)} {matrix2.GetElement(2,1)} {matrix2.GetElement(2,2)} {matrix2.GetElement(2,3)}\n{matrix2.GetElement(3,0)} {matrix2.GetElement(3,1)} {matrix2.GetElement(3,2)} {matrix2.GetElement(3,3)}\n\n'
          file.write(f'5dof_target1_rgb_{timestamp_millis}')
          file.write(matrixString2)
      if self.transformSelector3.currentNode(): 
        with open(f'{self.imagePathBox.text}/log.txt', 'a+') as file:
          matrix3 = vtk.vtkMatrix4x4()
          self.transformSelector3.currentNode().GetMatrixTransformToWorld(matrix3)
          matrixString3 = f'\n{matrix3.GetElement(0,0)} {matrix3.GetElement(0,1)} {matrix3.GetElement(0,2)} {matrix3.GetElement(0,3)}\n{matrix3.GetElement(1,0)} {matrix3.GetElement(1,1)} {matrix3.GetElement(1,2)} {matrix3.GetElement(1,3)}\n{matrix3.GetElement(2,0)} {matrix3.GetElement(2,1)} {matrix3.GetElement(2,2)} {matrix3.GetElement(2,3)}\n{matrix3.GetElement(3,0)} {matrix3.GetElement(3,1)} {matrix3.GetElement(3,2)} {matrix3.GetElement(3,3)}\n\n'
          file.write(f'5dof_target2_rgb_{timestamp_millis}')
          file.write(matrixString3)
      if self.transformSelector4.currentNode(): 
        with open(f'{self.imagePathBox.text}/log.txt', 'a+') as file:
          matrix4 = vtk.vtkMatrix4x4()
          self.transformSelector4.currentNode().GetMatrixTransformToWorld(matrix4)
          matrixString4 = f'\n{matrix4.GetElement(0,0)} {matrix4.GetElement(0,1)} {matrix4.GetElement(0,2)} {matrix4.GetElement(0,3)}\n{matrix4.GetElement(1,0)} {matrix4.GetElement(1,1)} {matrix4.GetElement(1,2)} {matrix4.GetElement(1,3)}\n{matrix4.GetElement(2,0)} {matrix4.GetElement(2,1)} {matrix4.GetElement(2,2)} {matrix4.GetElement(2,3)}\n{matrix4.GetElement(3,0)} {matrix4.GetElement(3,1)} {matrix4.GetElement(3,2)} {matrix4.GetElement(3,3)}\n\n'
          file.write(f'5dof_target3_rgb_{timestamp_millis}')
          file.write(matrixString4)
      if self.transformSelector5.currentNode(): 
        with open(f'{self.imagePathBox.text}/log.txt', 'a+') as file:
          matrix5 = vtk.vtkMatrix4x4()
          self.transformSelector5.currentNode().GetMatrixTransformToWorld(matrix5)
          matrixString5 = f'\n{matrix5.GetElement(0,0)} {matrix5.GetElement(0,1)} {matrix5.GetElement(0,2)} {matrix5.GetElement(0,3)}\n{matrix5.GetElement(1,0)} {matrix5.GetElement(1,1)} {matrix5.GetElement(1,2)} {matrix5.GetElement(1,3)}\n{matrix5.GetElement(2,0)} {matrix5.GetElement(2,1)} {matrix5.GetElement(2,2)} {matrix5.GetElement(2,3)}\n{matrix5.GetElement(3,0)} {matrix5.GetElement(3,1)} {matrix5.GetElement(3,2)} {matrix5.GetElement(3,3)}\n\n'
          file.write(f'5dof_target4_rgb_{timestamp_millis}')
          file.write(matrixString5)
        


class EndoscopeRecordLogic:
  def __init__(self):
    pass

