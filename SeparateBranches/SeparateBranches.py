import os
from pyexpat import model
import unittest
# from matplotlib.pyplot import get
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import numpy as np
import math
from sys import platform
from Resources import layers

class SeparateBranches(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Bakse/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Separate Branches"
    self.parent.categories = ["Navigation"]
    self.parent.dependencies = []
    self.parent.contributors = ["Franklin King"]
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

class SeparateBranchesWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)

  def cleanup(self):
    pass

  def onReload(self,moduleName="SeparateBranches"):
    globals()[moduleName] = slicer.util.reloadScriptedModule(moduleName)    

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Depth map point cloud collapsible button
    self.IOCollapsibleButton = ctk.ctkCollapsibleButton()
    self.IOCollapsibleButton.text = "I/O"
    self.IOCollapsibleButton.collapsed = False
    self.layout.addWidget(self.IOCollapsibleButton)
    IOLayout = qt.QFormLayout(self.IOCollapsibleButton)    

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
    IOLayout.addRow("Center Line Model: ", self.centerlineSelector)
    
    self.runButton = qt.QPushButton("Separate")
    IOLayout.addRow(self.runButton)
    self.runButton.connect('clicked()', self.onRun)
    
    self.inputsFiducialSelector = slicer.qMRMLNodeComboBox()
    self.inputsFiducialSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
    self.inputsFiducialSelector.selectNodeUponCreation = False
    self.inputsFiducialSelector.noneEnabled = True
    self.inputsFiducialSelector.addEnabled = True
    self.inputsFiducialSelector.showHidden = False
    self.inputsFiducialSelector.setMRMLScene( slicer.mrmlScene )
    IOLayout.addRow("Fiducials:", self.inputsFiducialSelector)
    
    self.createFiducialsButton = qt.QPushButton("Create Fiducials")
    IOLayout.addRow(self.createFiducialsButton)
    self.createFiducialsButton.connect('clicked()', self.onCreateFiducials)
    
    self.indexBox1 = qt.QSpinBox()
    self.indexBox1.setSingleStep(1)
    self.indexBox1.setMaximum(10000)
    self.indexBox1.setMinimum(1)
    self.indexBox1.value = 1
    IOLayout.addRow("Index 1: ", self.indexBox1)
    
    self.indexBox2 = qt.QSpinBox()
    self.indexBox2.setSingleStep(1)
    self.indexBox2.setMaximum(10000)
    self.indexBox2.setMinimum(1)
    self.indexBox2.value = 1
    IOLayout.addRow("Index 2: ", self.indexBox2)
    
    self.createBranchButton = qt.QPushButton("Create Branch")
    IOLayout.addRow(self.createBranchButton)
    self.createBranchButton.connect('clicked()', self.onCreateBranch)

    # Add vertical spacer
    self.layout.addStretch(1)

  def find_duplicate_points(self, polydata):
    epsilon = 0.00001
    points = polydata.GetPoints()
    num_points = points.GetNumberOfPoints()
    duplicate_ids = []
    
    for i in range(num_points):
        point1 = points.GetPoint(i)
        for j in range(i + 1, num_points):
            point2 = points.GetPoint(j)
            distance = vtk.vtkMath.Distance2BetweenPoints(point1, point2)
            if distance < epsilon ** 2:  # Compare squared distance to avoid computing square root
                duplicate_ids.append(i)
                duplicate_ids.append(j)
    
    unique_list = []
    for item in duplicate_ids:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list


  def separate_branches(self, polydata, branching_points):
    segments = []
    original_points = polydata.GetPoints()
    original_num_points = original_points.GetNumberOfPoints()
    
    for pointID in branching_points:
      points = vtk.vtkPoints()
      n = pointID
      while True:
        if n >= original_num_points:
          break
        newPoint1 = original_points.GetPoint(n)
        points.InsertNextPoint(newPoint1[0], newPoint1[1], newPoint1[2])
        n += 1
        if n in branching_points:
          newPoint2 = original_points.GetPoint(n)
          distance = math.sqrt(vtk.vtkMath.Distance2BetweenPoints(newPoint1, newPoint2))
          if distance < 10:
            points.InsertNextPoint(newPoint2[0], newPoint2[1], newPoint2[2])
          break
      
      lines = vtk.vtkCellArray()
      for i in range(points.GetNumberOfPoints() - 1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, i + 1)
        lines.InsertNextCell(line)
      
      new_polydata = vtk.vtkPolyData()
      new_polydata.SetPoints(points)
      new_polydata.SetLines(lines)
      
      model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
      model_node.SetAndObservePolyData(new_polydata)
      model_node.SetName("Branch")
      slicer.mrmlScene.AddNode(model_node)
      model_node.CreateDefaultDisplayNodes()
      model_node.GetModelDisplayNode().SetLineWidth(3)

  def onRun(self):
    polydata = self.centerlineSelector.currentNode().GetPolyData()
    branching_points = self.find_duplicate_points(polydata)
    self.separate_branches(polydata, branching_points)

  def onCreateFiducials(self):
    polydata = self.centerlineSelector.currentNode().GetPolyData()
    points = polydata.GetPoints()
    fiducialPoints = []
    for i in range(points.GetNumberOfPoints()):
        point = points.GetPoint(i)
        fiducialPoints.append(point)
    
    fiducialNode = self.inputsFiducialSelector.currentNode()
    for point in fiducialPoints:
        fiducialNode.AddFiducial(*point)
  
  def onCreateBranch(self):
    polydata = self.centerlineSelector.currentNode().GetPolyData()
    points = vtk.vtkPoints()
    index1 = self.indexBox1.value - 1
    index2 = self.indexBox2.value - 1
    
    original_points = polydata.GetPoints()
    for n in range(index1, index2+1):
      newPoint = original_points.GetPoint(n)
      points.InsertNextPoint(newPoint[0], newPoint[1], newPoint[2])
    
    lines = vtk.vtkCellArray()
    for i in range(points.GetNumberOfPoints() - 1):
      line = vtk.vtkLine()
      line.GetPointIds().SetId(0, i)
      line.GetPointIds().SetId(1, i + 1)
      lines.InsertNextCell(line)
    
    new_polydata = vtk.vtkPolyData()
    new_polydata.SetPoints(points)
    new_polydata.SetLines(lines)
    
    model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
    model_node.SetAndObservePolyData(new_polydata)
    model_node.SetName("Branch")
    slicer.mrmlScene.AddNode(model_node)
    model_node.CreateDefaultDisplayNodes()
    model_node.GetModelDisplayNode().SetLineWidth(3)
  