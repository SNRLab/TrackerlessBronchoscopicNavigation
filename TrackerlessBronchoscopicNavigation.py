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
    self.parent.categories = ["Filtering"]
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

  def cleanup(self):
    self.stepCount = 0

  def onReload(self,moduleName="TrackerlessBronchoscopicNavigation"):
    self.stepCount = 0
    globals()[moduleName] = slicer.util.reloadScriptedModule(moduleName)    

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # IO collapsible button
    self.IOCollapsibleButton = ctk.ctkCollapsibleButton()
    self.IOCollapsibleButton.text = "I/O"
    self.IOCollapsibleButton.collapsed = False
    self.layout.addWidget(self.IOCollapsibleButton)
    IOLayout = qt.QFormLayout(self.IOCollapsibleButton)

    self.inputImageSelector = slicer.qMRMLNodeComboBox()
    self.inputImageSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputImageSelector.selectNodeUponCreation = False
    self.inputImageSelector.noneEnabled = False
    self.inputImageSelector.addEnabled = True
    self.inputImageSelector.removeEnabled = True
    self.inputImageSelector.setMRMLScene(slicer.mrmlScene)
    IOLayout.addRow("Camera Image: ", self.inputImageSelector)

    self.inputTransformSelector = slicer.qMRMLNodeComboBox()
    self.inputTransformSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.inputTransformSelector.selectNodeUponCreation = False
    self.inputTransformSelector.noneEnabled = False
    self.inputTransformSelector.addEnabled = True
    self.inputTransformSelector.removeEnabled = True
    self.inputImageSelector.setMRMLScene(slicer.mrmlScene)
    IOLayout.addRow("Camera Transform: ", self.inputTransformSelector)    

    # Step mode collapsible button
    self.stepModeCollapsibleButton = ctk.ctkCollapsibleButton()
    self.stepModeCollapsibleButton.text = "Step Mode"
    self.stepModeCollapsibleButton.collapsed = False
    self.layout.addWidget(self.stepModeCollapsibleButton)
    stepModeLayout = qt.QFormLayout(self.stepModeCollapsibleButton)

    self.outputPathBox = qt.QLineEdit("D:/MeghaData/predictions/dataset_14/keyframe_1")
    self.outputPathBox.setReadOnly(True)
    self.outputPathButton = qt.QPushButton("...")
    self.outputPathButton.clicked.connect(self.select_directory)
    pathBoxLayout = qt.QHBoxLayout()
    pathBoxLayout.addWidget(self.outputPathBox)
    pathBoxLayout.addWidget(self.outputPathButton)
    stepModeLayout.addRow(pathBoxLayout)

    self.stepButton = qt.QPushButton("Step Image")
    stepModeLayout.addWidget(self.unwrapImageButton)
    self.stepButton.connect('clicked()', self.onStepImage)

    self.stepLabel = qt.QLabel("0")
    stepModeLayout.addWidget(self.stepLabel)

    self.resetStepButton = qt.QPushButton("Reset Step Count")
    stepModeLayout.addWidget(self.unwrapImageButton)
    self.stepButton.connect('clicked()', self.onResetStepCount)

  def select_directory(self):
    directory = qt.QFileDialog.getExistingDirectory(self.parent, "Select Directory")
    if directory:
      self.outputPathBox.setText(directory)

  def onResetStepCount(self):
    self.stepCount = 0

  def onStepImage(self):
    import re
    self.stepCount += 1
    stepCountString = str(self.stepCount)
    self.stepLabel.setText(stepCountString)
    
    prefix = "frame_data"
    imageFilename = ""
    frameDataFilename = ""
    for filename in os.listdir(self.outputPathBox.text):
      if filename.lstrip('0').split('.')[0] == stepCountString:
        imageFilename = filename
      
      if filename.startswith(prefix):
        stripped_filename = re.sub('^' + prefix, '', filename).lstrip('0').split('.')[0]
        if stripped_filename == stepCountString:
          frameDataFilename = filename
        
    # Display image
    image = Image.open(f'{self.outputPathBox.text}/{imageFilename}')
    image_array = np.array(image)

    vtk_image_data = vtk.vtkImageData()
    vtk_image_data.SetDimensions(image_array.shape[1], image_array.shape[0], 1)
    vtk_image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    # Convert the numpy array to a VTK array and set it as the scalars for the vtkImageData object
    vtk_array = numpy_support.numpy_to_vtk(num_array=image_array.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_image_data.GetPointData().SetScalars(vtk_array)

    # Set the vtkImageData object as the image data for the volume node
    self.inputImageSelector.currentNode().SetAndObserveImageData(vtk_image_data)
    
    