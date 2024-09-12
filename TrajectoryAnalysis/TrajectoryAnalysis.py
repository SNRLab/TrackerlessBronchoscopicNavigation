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
import statistics
import csv

class TrajectoryAnalysis(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Bakse/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Trajectory Analysis"
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

class TrajectoryAnalysisWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)

  def cleanup(self):
    pass

  def onReload(self,moduleName="TrajectoryAnalysis"):
    globals()[moduleName] = slicer.util.reloadScriptedModule(moduleName)    

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    self.DistanceCalculationCollapsibleButton = ctk.ctkCollapsibleButton()
    self.DistanceCalculationCollapsibleButton.text = "Distance Calculation"
    self.DistanceCalculationCollapsibleButton.collapsed = False
    self.layout.addWidget(self.DistanceCalculationCollapsibleButton)
    DistanceLayout = qt.QFormLayout(self.DistanceCalculationCollapsibleButton)

    self.pathBox = qt.QLineEdit("D:/Partners HealthCare Dropbox/Franklin King/SNRLabDisk/Projects/CanonProj/TrackerlessNavigation/ExperimentResults/Model_Results/BoxPhantom1/Analysis")
    self.pathBox.setReadOnly(False)
    self.pathButton = qt.QPushButton("...")
    self.pathButton.clicked.connect(self.select_directory)
    pathBoxLayout = qt.QHBoxLayout()
    pathBoxLayout.addWidget(self.pathBox)
    pathBoxLayout.addWidget(self.pathButton)
    DistanceLayout.addRow("Output Folder: ", pathBoxLayout)

    self.stepSkipBox = qt.QSpinBox()
    self.stepSkipBox.setSingleStep(1)
    self.stepSkipBox.setMaximum(100)
    self.stepSkipBox.setMinimum(1)
    self.stepSkipBox.value = 3
    DistanceLayout.addRow("Step skip: ", self.stepSkipBox)

    textEditLabel = qt.QLabel("Any trajectories not found will be ignored")
    DistanceLayout.addRow(textEditLabel)

    self.gtTextBox = qt.QTextEdit()
    gtDefaultNames = ['GT_1', 'GT_2', 'GT_3', 'GT_4', 'GT_5', 'GT_6', 'GT_7', 'GT_8']
    self.gtTextBox.setPlainText("\n".join(gtDefaultNames))
    DistanceLayout.addRow("GT Trajectories: ", self.gtTextBox)

    self.dataTextBox = qt.QTextEdit()
    dataDefaultNames = ['A_1',
        'A_2',
        'A_3',
        'A_4',
        'A_5',
        'A_6',
        'A_7',
        'A_8',
        'B_1_PoseOnly',
        'B_1_CenterCorrection',
        'B_2_PoseOnly',
        'B_2_CenterCorrection',
        'B_3_PoseOnly',
        'B_3_CenterCorrection',
        'B_4_PoseOnly',
        'B_4_CenterCorrection',
        'B_5_PoseOnly',
        'B_5_CenterCorrection',
        'B_6_PoseOnly',
        'B_6_CenterCorrection',
        'B_7_PoseOnly',
        'B_7_CenterCorrection',
        'B_8_PoseOnly',
        'B_8_CenterCorrection',
        'C_1_PoseOnly',
        'C_1_CenterCorrection',
        'C_2_PoseOnly',
        'C_2_CenterCorrection',
        'C_3_PoseOnly',
        'C_3_CenterCorrection',
        'C_4_PoseOnly',
        'C_4_CenterCorrection',
        'C_5_PoseOnly',
        'C_5_CenterCorrection',
        'C_6_PoseOnly',
        'C_6_CenterCorrection',
        'C_7_PoseOnly',
        'C_7_CenterCorrection',
        'C_8_PoseOnly',
        'C_8_CenterCorrection',
        'D_1_PoseOnly',
        'D_1_CenterCorrection',
        'D_2_PoseOnly',
        'D_2_CenterCorrection',
        'D_3_PoseOnly',
        'D_3_CenterCorrection',
        'D_4_PoseOnly',
        'D_4_CenterCorrection',
        'D_5_PoseOnly',
        'D_5_CenterCorrection',
        'D_6_PoseOnly',
        'D_6_CenterCorrection',
        'D_7_PoseOnly',
        'D_7_CenterCorrection',
        'D_8_PoseOnly',
        'D_8_CenterCorrection',
        'E_1_PoseOnly',
        'E_1_CenterCorrection',
        'E_2_PoseOnly',
        'E_2_CenterCorrection',
        'E_3_PoseOnly',
        'E_3_CenterCorrection',
        'E_4_PoseOnly',
        'E_4_CenterCorrection',
        'E_5_PoseOnly',
        'E_5_CenterCorrection',
        'E_6_PoseOnly',
        'E_6_CenterCorrection',
        'E_7_PoseOnly',
        'E_7_CenterCorrection',
        'E_8_PoseOnly',
        'E_8_CenterCorrection',
        'F_1_PoseOnly',
        'F_1_CenterCorrection',
        'F_2_PoseOnly',
        'F_2_CenterCorrection',
        'F_3_PoseOnly',
        'F_3_CenterCorrection',
        'F_4_PoseOnly',
        'F_4_CenterCorrection',
        'F_5_PoseOnly',
        'F_5_CenterCorrection',
        'F_6_PoseOnly',
        'F_6_CenterCorrection',
        'F_7_PoseOnly',
        'F_7_CenterCorrection',
        'F_8_PoseOnly',
        'F_8_CenterCorrection',
        'G_1_PoseOnly',
        'G_1_CenterCorrection',
        'G_2_PoseOnly',
        'G_2_CenterCorrection',
        'G_3_PoseOnly',
        'G_3_CenterCorrection',
        'G_4_PoseOnly',
        'G_4_CenterCorrection',
        'G_5_PoseOnly',
        'G_5_CenterCorrection',
        'G_6_PoseOnly',
        'G_6_CenterCorrection',
        'G_7_PoseOnly',
        'G_7_CenterCorrection',
        'G_8_PoseOnly',
        'G_8_CenterCorrection']
    self.dataTextBox.setPlainText("\n".join(dataDefaultNames))
    DistanceLayout.addRow("Data Trajectories: ", self.dataTextBox)

    self.calculateDistancesButton = qt.QPushButton("Calculate Distances")
    DistanceLayout.addRow(self.calculateDistancesButton)
    self.calculateDistancesButton.connect('clicked()', self.onCalculateDistances)
    
    self.calculateLabelDistancesButton = qt.QPushButton("Calculate Label Distances")
    DistanceLayout.addRow(self.calculateLabelDistancesButton)
    self.calculateLabelDistancesButton.connect('clicked()', self.onCalculateLabelDistances)    

    self.LabelCalculationCollapsibleButton = ctk.ctkCollapsibleButton()
    self.LabelCalculationCollapsibleButton.text = "Label Calculation"
    self.LabelCalculationCollapsibleButton.collapsed = False
    self.layout.addWidget(self.LabelCalculationCollapsibleButton)
    LabelLayout = qt.QFormLayout(self.LabelCalculationCollapsibleButton)

    self.labelFiducialsButton = qt.QPushButton("Label All Fiducial Lists with Branches")
    LabelLayout.addRow(self.labelFiducialsButton)
    self.labelFiducialsButton.connect('clicked()', self.onLabelFiducials)

    self.calculateLabelsButton = qt.QPushButton("Calculate Label Comparison")
    LabelLayout.addRow(self.calculateLabelsButton)
    self.calculateLabelsButton.connect('clicked()', self.onCalculateLabels)

    # Add vertical spacer
    self.layout.addStretch(1)


  def select_directory(self):
    directory = qt.QFileDialog.getExistingDirectory(self.parent, "Select Directory")
    if directory:
      self.pathBox.setText(directory)


  def calculate_distance(self, fiducial1, fiducial2):
    return np.sqrt((fiducial1[0]-fiducial2[0])**2 + (fiducial1[1]-fiducial2[1])**2 + (fiducial1[2]-fiducial2[2])**2)


  def get_fiducials(self, fiducial_node):
    if fiducial_node:
      num_fiducials = fiducial_node.GetNumberOfFiducials()
      fiducials = []
      for i in range(num_fiducials):
        coord = [0, 0, 0]
        fiducial_node.GetNthFiducialPosition(i, coord)
        fiducials.append(tuple(coord))
      return fiducials
    else:
      return []


  def write_distances(self, listA, listB, filename, skip=3):
    totalDistance = 0
    distances = []
    distanceValues = []
    for i in range(min(len(listA),len(listB))):
      distance = self.calculate_distance(listA[i*skip], listB[i])
      totalDistance += distance
      distances.append([(i*skip)+1, str(distance)])
      distanceValues.append(distance)
    mean = totalDistance / min(len(listA),len(listB))
    std_dev = statistics.stdev(distanceValues)
    startLines = [[f'Mean: ', mean],[f'Std_Dev: ', std_dev], ["Frame", "Error (mm)"]]
    filePath = self.pathBox.text + "/" + filename

    with open((filePath), mode='w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(startLines)
    with open((filePath), mode='a', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(distances)


  def onCalculateDistances(self):
    gtText = self.gtTextBox.toPlainText()
    GT_Paths = gtText.split("\n")
    dataText = self.dataTextBox.toPlainText()
    Paths = dataText.split("\n")

    for GT_Path in GT_Paths:
      try: 
        gt_node = slicer.util.getNode(GT_Path)
      except:
        gt_node = None

      listA = self.get_fiducials(gt_node)
      
      if len(listA) > 0:
        print(f'GT: {gt_node.GetName()}')

        for Path in Paths:
          if int(Path.split('_')[1]) == int(GT_Path.split('_')[1]):
            try: 
              node = slicer.util.getNode(Path)
            except:
              node = None
            listB = self.get_fiducials(node)

            if len(listB) > 0:
              print(f'Writing {Path}')
              filename = f'{GT_Path}-{Path}.csv'
              self.write_distances(listA, listB, filename, skip=self.stepSkipBox.value)
            else:
              print(f'Skipped {Path}')
              

  def onCalculateLabelDistances(self):
    gtText = self.gtTextBox.toPlainText()
    GT_Paths = gtText.split("\n")
    dataText = self.dataTextBox.toPlainText()
    Paths = dataText.split("\n")

    for GT_Path in GT_Paths:
      try: 
        gt_node = slicer.util.getNode(f'{GT_Path}_ClosestCenterline')
      except:
        gt_node = None

      listA = self.get_fiducials(gt_node)
      
      if len(listA) > 0:
        print(f'GT: {gt_node.GetName()}')

        for Path in Paths:
          if int(Path.split('_')[1]) == int(GT_Path.split('_')[1]):
            try: 
              node = slicer.util.getNode(f'{Path}_ClosestCenterline')
            except:
              node = None
            listB = self.get_fiducials(node)

            if len(listB) > 0:
              print(f'Writing {Path}')
              filename = f'{GT_Path}-{Path}-LabelDistance.csv'
              self.write_distances(listA, listB, filename, skip=self.stepSkipBox.value)
            else:
              print(f'Skipped {Path}')              


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


  def onLabelFiducials(self):
    # Find branches
    subjectHierarchyNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    branchesID = subjectHierarchyNode.GetItemByName("Branches")
    branchItemIDs = vtk.vtkIdList()
    subjectHierarchyNode.GetItemChildren(branchesID, branchItemIDs, 'vtkMRMLModelNode')
    branchNodes = []
    for i in range(branchItemIDs.GetNumberOfIds()):
      branchItemID = branchItemIDs.GetId(i)
      branchNode = subjectHierarchyNode.GetItemDataNode(branchItemID)
      branchNodes.append(branchNode)

    gtText = self.gtTextBox.toPlainText()
    GT_Paths = gtText.split("\n")
    dataText = self.dataTextBox.toPlainText()
    Paths = dataText.split("\n")

    for GT_Path in GT_Paths:
      try: 
        gt_node = slicer.util.getNode(GT_Path)
      except:
        gt_node = None
      
      if gt_node:
        closestCenterlineFiducialListNode = slicer.util.getNode(f'{gt_node.GetName()}_ClosestCenterline')
        closestCenterlineFiducialListNode.RemoveAllControlPoints()

        for i in range(gt_node.GetNumberOfFiducials()):
          fiducialCoords = [0.0, 0.0, 0.0]
          gt_node.GetNthFiducialPosition(i, fiducialCoords)

          closestModelNode = None
          minDistance = float('inf')
          closestPoint = [0,0,0]

          for branchNode in branchNodes:
            closestPointOnLine = self.findClosestPointOnLine(branchNode, fiducialCoords)
            distance = abs(np.linalg.norm(np.array(closestPointOnLine) - np.array(fiducialCoords)))
            if distance < minDistance:
              minDistance = distance
              closestPoint = closestPointOnLine
              closestModelNode = branchNode
          
          # Rename the fiducial
          if closestModelNode:
            gt_node.SetNthFiducialLabel(i, closestModelNode.GetName())
            closestCenterlineFiducialListNode.AddFiducial(closestPoint[0],closestPoint[1],closestPoint[2], closestModelNode.GetName())

    for Path in Paths:
      try: 
        node = slicer.util.getNode(Path)
      except:
        node = None
      
      if node:
        closestCenterlineFiducialListNode = slicer.util.getNode(f'{node.GetName()}_ClosestCenterline')
        closestCenterlineFiducialListNode.RemoveAllControlPoints()

        for i in range(node.GetNumberOfFiducials()):
          fiducialCoords = [0.0, 0.0, 0.0]
          node.GetNthFiducialPosition(i, fiducialCoords)

          closestModelNode = None
          minDistance = float('inf')
          closestPoint = [0,0,0]

          for branchNode in branchNodes:
            closestPointOnLine = self.findClosestPointOnLine(branchNode, fiducialCoords)
            distance = abs(np.linalg.norm(np.array(closestPointOnLine) - np.array(fiducialCoords)))
            if distance < minDistance:
              minDistance = distance
              closestPoint = closestPointOnLine
              closestModelNode = branchNode
          
          # Rename the fiducial
          if closestModelNode:
            node.SetNthFiducialLabel(i, closestModelNode.GetName())
            closestCenterlineFiducialListNode.AddFiducial(closestPoint[0],closestPoint[1],closestPoint[2], closestModelNode.GetName())


  def write_labels(self, gtNode, node, filename, skip=3):
    labelBools = []
    for i in range(min(gtNode.GetNumberOfFiducials(), node.GetNumberOfFiducials())):
      sameName = False
      if gtNode.GetNthFiducialLabel(i) == node.GetNthFiducialLabel(i):
        sameName = True
      labelBools.append([(i*skip)+1, gtNode.GetNthFiducialLabel(i), node.GetNthFiducialLabel(i), str(sameName)])
    startLines = [["Frame", "GT Label", "Trajectory Label", "Same Label?"]]
    filePath = self.pathBox.text + "/" + filename

    with open((filePath), mode='w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(startLines)
    with open((filePath), mode='a', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(labelBools)


  def onCalculateLabels(self):
    gtText = self.gtTextBox.toPlainText()
    GT_Paths = gtText.split("\n")
    dataText = self.dataTextBox.toPlainText()
    Paths = dataText.split("\n")

    for GT_Path in GT_Paths:
      try: 
        gt_node = slicer.util.getNode(GT_Path)
      except:
        gt_node = None
      
      if gt_node and gt_node.GetNumberOfFiducials() > 0:
        print(f'GT: {gt_node.GetName()}')

        for Path in Paths:
          if int(Path.split('_')[1]) == int(GT_Path.split('_')[1]):
            try: 
              node = slicer.util.getNode(Path)
            except:
              node = None

            if node and node.GetNumberOfFiducials() > 0:
              print(f'Writing {Path}')
              filename = f'{GT_Path}-{Path}-Labels.csv'
              self.write_labels(gt_node, node, filename, skip=self.stepSkipBox.value)
            else:
              print(f'Skipped {Path}')
