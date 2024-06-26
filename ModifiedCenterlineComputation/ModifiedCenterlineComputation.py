# slicer imports
import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from vtk.util.numpy_support import vtk_to_numpy
import logging
import numpy as np

# python includes
import math

#
# Centerline Computation using VMTK based Tools
#

class ModifiedCenterlineComputation(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Modified Centerline Computation"
    self.parent.categories = ["Vascular Modeling Toolkit"]
    self.parent.dependencies = []
    self.parent.contributors = ["Daniel Haehn (Boston Children's Hospital)", "Luca Antiga (Orobix)", "Steve Pieper (Isomics)", "Andras Lasso (PerkLab)"]
    self.parent.helpText = """
"""
    self.parent.acknowledgementText = """
""" # replace with organization, grant and thanks.

class ModifiedCenterlineComputationWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)

    # the pointer to the logic
    self.logic = CenterlineComputationLogic()

    if not parent:
      # after setup, be ready for events
      self.parent.show()
    else:
      # register default slots
      self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)', self.onMRMLSceneChanged)

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    #
    # Inputs
    #

    inputsCollapsibleButton = ctk.ctkCollapsibleButton()
    inputsCollapsibleButton.text = "Inputs"
    self.layout.addWidget(inputsCollapsibleButton)
    inputsFormLayout = qt.QFormLayout(inputsCollapsibleButton)

    # inputVolume selector
    self.inputModelNodeSelector = slicer.qMRMLNodeComboBox()
    self.inputModelNodeSelector.objectName = 'inputModelNodeSelector'
    self.inputModelNodeSelector.toolTip = "Select the input model."
    self.inputModelNodeSelector.nodeTypes = ['vtkMRMLModelNode']
    self.inputModelNodeSelector.hideChildNodeTypes = ['vtkMRMLAnnotationNode']  # hide all annotation nodes
    self.inputModelNodeSelector.noneEnabled = False
    self.inputModelNodeSelector.addEnabled = False
    self.inputModelNodeSelector.removeEnabled = False
    inputsFormLayout.addRow("Vessel tree model:", self.inputModelNodeSelector)
    self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)',
                        self.inputModelNodeSelector, 'setMRMLScene(vtkMRMLScene*)')

    # seed selector
    self.seedFiducialsNodeSelector = slicer.qSlicerSimpleMarkupsWidget()
    self.seedFiducialsNodeSelector.objectName = 'seedFiducialsNodeSelector'
    self.seedFiducialsNodeSelector = slicer.qSlicerSimpleMarkupsWidget()
    self.seedFiducialsNodeSelector.objectName = 'seedFiducialsNodeSelector'
    self.seedFiducialsNodeSelector.toolTip = "Select a fiducial to use as the origin of the Centerline."
    self.seedFiducialsNodeSelector.setNodeBaseName("OriginSeed")
    self.seedFiducialsNodeSelector.defaultNodeColor = qt.QColor(0,255,0)
    self.seedFiducialsNodeSelector.tableWidget().hide()
    self.seedFiducialsNodeSelector.markupsSelectorComboBox().noneEnabled = False
    self.seedFiducialsNodeSelector.markupsPlaceWidget().placeMultipleMarkups = slicer.qSlicerMarkupsPlaceWidget.ForcePlaceSingleMarkup
    inputsFormLayout.addRow("Start point:", self.seedFiducialsNodeSelector)
    self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)',
                        self.seedFiducialsNodeSelector, 'setMRMLScene(vtkMRMLScene*)')


    # ROI selector for path tracing from origin to ROI
    self.roiFiducialsNodeSelector = slicer.qSlicerSimpleMarkupsWidget()
    self.roiFiducialsNodeSelector.objectName = 'roiFiducialsNodeSelector'
    self.roiFiducialsNodeSelector = slicer.qSlicerSimpleMarkupsWidget()
    self.roiFiducialsNodeSelector.objectName = 'roiFiducialsNodeSelector'
    self.roiFiducialsNodeSelector.toolTip = "Select a fiducial to use as the target region of interest."
    self.roiFiducialsNodeSelector.setNodeBaseName("ROISeed")
    self.roiFiducialsNodeSelector.defaultNodeColor = qt.QColor(255,0,0)
    self.roiFiducialsNodeSelector.tableWidget().hide()
    self.roiFiducialsNodeSelector.markupsSelectorComboBox().noneEnabled = False
    self.roiFiducialsNodeSelector.markupsPlaceWidget().placeMultipleMarkups = slicer.qSlicerMarkupsPlaceWidget.ForcePlaceSingleMarkup
    inputsFormLayout.addRow("Target region of interest (ROI):", self.roiFiducialsNodeSelector)
    self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)',
                        self.roiFiducialsNodeSelector, 'setMRMLScene(vtkMRMLScene*)')

    self.colorByRadiusCheckbox = qt.QCheckBox()
    self.colorByRadiusCheckbox.toolTip = "Toggle whether or not to overlay a radius colormap onto the centerline."
    inputsFormLayout.addRow("Color by radius: ", self.colorByRadiusCheckbox)

    self.colorByLocalCurvatureCheckbox = qt.QCheckBox()
    self.colorByLocalCurvatureCheckbox.toolTip = "Toggle whether or not to overlay a local curvature colormap onto the centerline."
    inputsFormLayout.addRow("Color by local curvature: ", self.colorByLocalCurvatureCheckbox)

    self.colorByGlobalRelativeAngleCheckbox = qt.QCheckBox()
    self.colorByGlobalRelativeAngleCheckbox.toolTip = "Toggle whether or not to overlay a global relative angle colormap onto the centerline."
    inputsFormLayout.addRow("Color by global relative angle: ", self.colorByGlobalRelativeAngleCheckbox)

    self.colorByPlaneRotationCheckbox = qt.QCheckBox()
    self.colorByPlaneRotationCheckbox.toolTip = "Toggle whether or not to overlay a plane rotation colormap onto the centerline."
    inputsFormLayout.addRow("Color by plane rotation: ", self.colorByPlaneRotationCheckbox)

    self.colorByCurvatureRateCheckbox = qt.QCheckBox()
    self.colorByCurvatureRateCheckbox.toolTip = "Toggle whether or not to overlay a global relative angle colormap onto the centerline."
    inputsFormLayout.addRow("Color by rate of curvature: ", self.colorByCurvatureRateCheckbox)

    self.colorByTotalIndexCheckbox = qt.QCheckBox()
    self.colorByTotalIndexCheckbox.toolTip = "Toggle whether or not to overlay a colormap of the total difficulty index onto the centerline."
    inputsFormLayout.addRow("Color by Total Difficulty Index (TDI): ", self.colorByTotalIndexCheckbox)

    self.colorByCumulativeIndexCheckbox = qt.QCheckBox()
    self.colorByCumulativeIndexCheckbox.toolTip = "Toggle whether or not to overlay a colormap of the cumulative difficulty index onto the centerline."
    inputsFormLayout.addRow("Color by Cumulative Difficulty Index (CDI): ", self.colorByCumulativeIndexCheckbox)

    # Allow the user to input scalar multiplier values for the threshold, default to 1.0 for all of them
    textLine = qt.QLabel()
    textLine.setText("Indicate scalar multiplier values for Total/Cumulative Index calculations:")
    inputsFormLayout.addWidget(textLine)

    # Radius
    self.radiusScalarTextbox = qt.QLineEdit("1.0")
    self.radiusScalarTextbox.setReadOnly(False)
    self.radiusScalarTextbox.setFixedWidth(75)
    inputsFormLayout.addRow("Radius scalar multiplier:", self.radiusScalarTextbox)

    # Local curvature
    self.localCurvatureScalarTextbox = qt.QLineEdit("1.0")
    self.localCurvatureScalarTextbox.setReadOnly(False)
    self.localCurvatureScalarTextbox.setFixedWidth(75)
    inputsFormLayout.addRow("Local curvature scalar multiplier:", self.localCurvatureScalarTextbox)

    # Global relative angle 
    self.angleScalarTextbox = qt.QLineEdit("1.0")
    self.angleScalarTextbox.setReadOnly(False)
    self.angleScalarTextbox.setFixedWidth(75)
    inputsFormLayout.addRow("Global relevative angle scalar multiplier:", self.angleScalarTextbox)

    # Curvature rate
    self.curvatureRateScalarTextbox = qt.QLineEdit("1.0")
    self.curvatureRateScalarTextbox.setReadOnly(False)
    self.curvatureRateScalarTextbox.setFixedWidth(75)
    inputsFormLayout.addRow("Rate of curvature scalar multiplier:", self.curvatureRateScalarTextbox)

    # Use a QButton group to make checkboxes exclusive
    self.group = qt.QButtonGroup()
    self.group.addButton(self.colorByRadiusCheckbox)
    self.group.addButton(self.colorByLocalCurvatureCheckbox)
    self.group.addButton(self.colorByGlobalRelativeAngleCheckbox)
    self.group.addButton(self.colorByPlaneRotationCheckbox)
    self.group.addButton(self.colorByCurvatureRateCheckbox)
    self.group.addButton(self.colorByTotalIndexCheckbox)
    self.group.addButton(self.colorByCumulativeIndexCheckbox)

    #
    # Outputs
    #

    outputsCollapsibleButton = ctk.ctkCollapsibleButton()
    outputsCollapsibleButton.text = "Outputs"
    self.layout.addWidget(outputsCollapsibleButton)
    outputsFormLayout = qt.QFormLayout(outputsCollapsibleButton)
                        
    # outputModel selector
    self.outputModelNodeSelector = slicer.qMRMLNodeComboBox()
    self.outputModelNodeSelector.objectName = 'outputModelNodeSelector'
    self.outputModelNodeSelector.toolTip = "Select the output model for the Centerlines."
    self.outputModelNodeSelector.nodeTypes = ['vtkMRMLModelNode']
    self.outputModelNodeSelector.baseName = "CenterlineComputationModel"
    self.outputModelNodeSelector.hideChildNodeTypes = ['vtkMRMLAnnotationNode']  # hide all annotation nodes
    self.outputModelNodeSelector.noneEnabled = True
    self.outputModelNodeSelector.noneDisplay = "Create new model"
    self.outputModelNodeSelector.addEnabled = True
    self.outputModelNodeSelector.selectNodeUponCreation = True
    self.outputModelNodeSelector.removeEnabled = True
    outputsFormLayout.addRow("Centerline model:", self.outputModelNodeSelector)
    self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)',
                        self.outputModelNodeSelector, 'setMRMLScene(vtkMRMLScene*)')

    self.outputEndPointsNodeSelector = slicer.qMRMLNodeComboBox()
    self.outputEndPointsNodeSelector.objectName = 'outputEndPointsNodeSelector'
    self.outputEndPointsNodeSelector.toolTip = "Select the output model for the Centerlines."
    self.outputEndPointsNodeSelector.nodeTypes = ['vtkMRMLMarkupsFiducialNode']
    self.outputEndPointsNodeSelector.baseName = "Centerline endpoints"
    self.outputEndPointsNodeSelector.noneEnabled = True
    self.outputEndPointsNodeSelector.noneDisplay = "Create new markups fiducial"
    self.outputEndPointsNodeSelector.addEnabled = True
    self.outputEndPointsNodeSelector.selectNodeUponCreation = True
    self.outputEndPointsNodeSelector.removeEnabled = True
    outputsFormLayout.addRow("Centerline endpoints:", self.outputEndPointsNodeSelector)
    self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)',
                        self.outputEndPointsNodeSelector, 'setMRMLScene(vtkMRMLScene*)')
                        
    # voronoiModel selector
    self.voronoiModelNodeSelector = slicer.qMRMLNodeComboBox()
    self.voronoiModelNodeSelector.objectName = 'voronoiModelNodeSelector'
    self.voronoiModelNodeSelector.toolTip = "Select the output model for the Voronoi Diagram."
    self.voronoiModelNodeSelector.nodeTypes = ['vtkMRMLModelNode']
    self.voronoiModelNodeSelector.baseName = "VoronoiModel"
    self.voronoiModelNodeSelector.hideChildNodeTypes = ['vtkMRMLAnnotationNode']  # hide all annotation nodes
    self.voronoiModelNodeSelector.noneEnabled = True
    self.voronoiModelNodeSelector.addEnabled = True
    self.voronoiModelNodeSelector.selectNodeUponCreation = True
    self.voronoiModelNodeSelector.removeEnabled = True
    outputsFormLayout.addRow("Voronoi Model:", self.voronoiModelNodeSelector)
    self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)',
                        self.voronoiModelNodeSelector, 'setMRMLScene(vtkMRMLScene*)')

    # Output Directory selector
    self.outputDirectory = ''
    self.outputDirectoryButton = qt.QPushButton('Select Output Directory')
    self.outputDirectoryButton.toolTip = "Click to change the output directory."
    self.outputDirectoryButton.connect("clicked()", self.onOutputDirectoryClicked)
    outputsFormLayout.addRow("Output Directory:", self.outputDirectoryButton)

    # Output filename
    self.outputFilenameTextbox = qt.QLineEdit("")
    self.outputFilenameTextbox.setReadOnly(False)
    self.outputFilenameTextbox.setFixedWidth(200)
    outputsFormLayout.addRow("Output File Name (ex. Case4_CDI):", self.outputFilenameTextbox)
    
    # Remove first point selector
    self.removeFirstPointCheckbox = qt.QCheckBox()
    self.removeFirstPointCheckbox.toolTip = "Toggle whether or not the first point should be removed."
    outputsFormLayout.addRow("Remove First Point From Output", self.removeFirstPointCheckbox)


    # Textboxes to output maximum values of calculated metrics
    self.minRadiusTextbox = qt.QLineEdit()
    self.minRadiusTextbox.setReadOnly(True)
    self.minRadiusTextbox.setFixedWidth(150)
    outputsFormLayout.addRow("Minimum Radius:", self.minRadiusTextbox)

    self.maxLocalCurvTextbox = qt.QLineEdit()
    self.maxLocalCurvTextbox.setReadOnly(True)
    self.maxLocalCurvTextbox.setFixedWidth(150)
    outputsFormLayout.addRow("Maximum Local Curvature:", self.maxLocalCurvTextbox)

    self.minAngleTextbox = qt.QLineEdit()
    self.minAngleTextbox.setReadOnly(True)
    self.minAngleTextbox.setFixedWidth(150)
    outputsFormLayout.addRow("Minimum Global Relative Angle:", self.minAngleTextbox)

    self.maxPlaneRotationTextbox = qt.QLineEdit()
    self.maxPlaneRotationTextbox.setReadOnly(True)
    self.maxPlaneRotationTextbox.setFixedWidth(150)
    outputsFormLayout.addRow("Maximum Plane Rotation Angle:", self.maxPlaneRotationTextbox)

    self.minPlaneRotationTextbox = qt.QLineEdit()
    self.minPlaneRotationTextbox.setReadOnly(True)
    self.minPlaneRotationTextbox.setFixedWidth(150)
    outputsFormLayout.addRow("Minimum Plane Rotation Angle:", self.minPlaneRotationTextbox)

    self.maxCurvRateTextbox = qt.QLineEdit()
    self.maxCurvRateTextbox.setReadOnly(True)
    self.maxCurvRateTextbox.setFixedWidth(150)
    outputsFormLayout.addRow("Maximum Rate of Curvature:", self.maxCurvRateTextbox)

    self.maxTotalDifficultyIndexTextbox = qt.QLineEdit()
    self.maxTotalDifficultyIndexTextbox.setReadOnly(True)
    self.maxTotalDifficultyIndexTextbox.setFixedWidth(150)
    outputsFormLayout.addRow("Maximum Total Difficulty Index:", self.maxTotalDifficultyIndexTextbox)

    self.maxCumulativeIndexTextbox = qt.QLineEdit()
    self.maxCumulativeIndexTextbox.setReadOnly(True)
    self.maxCumulativeIndexTextbox.setFixedWidth(150)
    outputsFormLayout.addRow("Final Cumulative Difficulty Index:", self.maxCumulativeIndexTextbox)



    #
    # Reset, preview and apply buttons
    #

    self.buttonBox = qt.QDialogButtonBox()
    self.previewButton = self.buttonBox.addButton(self.buttonBox.Discard)
    self.previewButton.setIcon(qt.QIcon())
    self.previewButton.text = "Preview"
    self.previewButton.toolTip = "Click to refresh the preview."
    self.startButton = self.buttonBox.addButton(self.buttonBox.Apply)
    self.startButton.setIcon(qt.QIcon())
    self.startButton.text = "Start"
    self.startButton.enabled = False
    self.startButton.toolTip = "Click to start the filtering."
    self.layout.addWidget(self.buttonBox)
    self.previewButton.connect("clicked()", self.onPreviewButtonClicked)
    self.startButton.connect("clicked()", self.onStartButtonClicked)

    self.inputModelNodeSelector.setMRMLScene(slicer.mrmlScene)
    self.seedFiducialsNodeSelector.setMRMLScene(slicer.mrmlScene)
    self.roiFiducialsNodeSelector.setMRMLScene(slicer.mrmlScene)
    self.outputModelNodeSelector.setMRMLScene(slicer.mrmlScene)
    self.outputEndPointsNodeSelector.setMRMLScene(slicer.mrmlScene)
    self.voronoiModelNodeSelector.setMRMLScene(slicer.mrmlScene)

    # compress the layout
    self.layout.addStretch(1)
    
  def onMRMLSceneChanged(self):
    logging.debug("onMRMLSceneChanged")

  def onOutputDirectoryClicked(self):
    fileDialog = qt.QFileDialog()
    self.outputDirectory = fileDialog.getExistingDirectory( None, 'Select Output Directory', self.outputDirectory )

  def onStartButtonClicked(self):
    qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
    # this is no preview
    self.start(False)
    qt.QApplication.restoreOverrideCursor()

  def onPreviewButtonClicked(self):
      # calculate the preview
      self.start(True)
      # activate startButton
      self.startButton.enabled = True

  def findClosestPointOnCenterline(self, point, allPts):
    # given a point in [x, y, z] format and a numpy array of all points on the centerline, return the pt on the centerline that is closest to that point
    
    print("we in it")

    min_dist = float("inf")
    current_point = allPts[0]
    # current_point = np.array([float(s) for s in (str( allPts[0] ).replace(' ','')[1:-1]).split(',')])
    print("current pt: ", current_point)
    for current_point in allPts:
      dist = np.linalg.norm(point-current_point)
      if dist < min_dist: new_closest = current_point
    return current_point

  def start(self, preview=False):
    logging.debug("Starting Centerline Computation..")

    # Determine scalar multiplier values from user input
    radiusScalar = float(self.radiusScalarTextbox.text)
    localCurvatureScalar = float(self.localCurvatureScalarTextbox.text)
    globalAngleScalar = float(self.angleScalarTextbox.text)
    curvatureRateScalar = float(self.curvatureRateScalarTextbox.text)

    outputFilename = self.outputFilenameTextbox.text

    # first we need the nodes
    currentModelNode = self.inputModelNodeSelector.currentNode()
    currentSeedsNode = self.seedFiducialsNodeSelector.currentNode()
    currentRoiNode = self.roiFiducialsNodeSelector.currentNode()
    currentOutputModelNode = self.outputModelNodeSelector.currentNode()
    currentEndPointsMarkupsNode = self.outputEndPointsNodeSelector.currentNode()
    currentVoronoiModelNode = self.voronoiModelNodeSelector.currentNode()

    if not currentModelNode:
      # we need a input volume node
      logging.error("Input model node required")
      return False

    if not currentSeedsNode:
      # we need a seeds node
      logging.error("Input seeds node required")
      return False

    # Set pathfinding mode if an ROI fiducial is selected
    if not currentRoiNode: pathfindingMode = False
    else: pathfindingMode = True


    if not currentOutputModelNode or currentOutputModelNode.GetID() == currentModelNode.GetID():
      # we need a current model node, the display node is created later
      newModelNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLModelNode")
      newModelNode.UnRegister(None)
      newModelNode.SetName(slicer.mrmlScene.GetUniqueNameByString(self.outputModelNodeSelector.baseName))
      currentOutputModelNode = slicer.mrmlScene.AddNode(newModelNode)
      currentOutputModelNode.CreateDefaultDisplayNodes()
      self.outputModelNodeSelector.setCurrentNode(currentOutputModelNode)

    if not currentEndPointsMarkupsNode or currentEndPointsMarkupsNode.GetID() == currentSeedsNode.GetID():
      # we need a current seed node, the display node is created later
      currentEndPointsMarkupsNode = slicer.mrmlScene.GetNodeByID(slicer.modules.markups.logic().AddNewFiducialNode("Centerline endpoints"))
      self.outputEndPointsNodeSelector.setCurrentNode(currentEndPointsMarkupsNode)

    # the output models
    preparedModel = vtk.vtkPolyData()
    print("preparedModel: ", preparedModel)
    model = vtk.vtkPolyData()
    network = vtk.vtkPolyData()
    voronoi = vtk.vtkPolyData()
    if pathfindingMode:
      pathfinding_network = vtk.vtkPolyData()
      pathfinding_voronoi = vtk.vtkPolyData()

    currentCoordinatesRAS = [0, 0, 0]

    # grab the current coordinates
    currentSeedsNode.GetNthFiducialPosition(0,currentCoordinatesRAS)

    # prepare the model
    preparedModel.DeepCopy(self.logic.prepareModel(currentModelNode.GetPolyData()))

    # decimate the model (only for network extraction)
    model.DeepCopy(self.logic.decimateSurface(preparedModel))

    # open the model at the seed (only for network extraction)
    model.DeepCopy(self.logic.openSurfaceAtPoint(model, currentCoordinatesRAS))

    # extract Network
    network.DeepCopy(self.logic.extractNetwork(model))

    #
    #
    # not preview mode: real computation!

    # # Computation of the centerline path between the start point and the ROI, if in pathfindingMode
    # if not preview and pathfindingMode:
    #   # clip surface at endpoints identified by the network extraction
    #   tupel = self.logic.clipSurfaceAtEndPoints(network, currentModelNode.GetPolyData())
    #   clippedSurface = tupel[0]

    #   # Write over endpoints array with ONLY the starting point and the indicated ROI point
    #   # Grab the coordinates of the ROI
    #   currentCoordinatesROI = [0, 0, 0]
    #   currentRoiNode.GetNthFiducialPosition(0,currentCoordinatesROI)
    #   print (currentCoordinatesROI)
    #   endpoints = vtk.vtkPoints()
    #   endpoints.InsertPoint(0, currentCoordinatesRAS) # Seed point
    #   endpoints.InsertPoint(1, currentCoordinatesROI) # Endpoint (ROIs)
    #   print(endpoints)

    #   # now find the one endpoint which is closest to the seed and use it as the source point for centerline computation
    #   # all other endpoints are the target points
    #   sourcePoint = [0, 0, 0]
    #   distancesToSeed = []
    #   targetPoints = []

    #   # we now need to loop through the endpoints two times
    #   # first loop is to detect the endpoint resulting in the tiny hole we poked in the surface
    #   # this is very close to our seed but not the correct sourcePoint
    #   for i in range(endpoints.GetNumberOfPoints()):
    #     currentPoint = endpoints.GetPoint(i)
    #     # get the euclidean distance
    #     currentDistanceToSeed = math.sqrt(math.pow((currentPoint[0] - currentCoordinatesRAS[0]), 2) +
    #                                        math.pow((currentPoint[1] - currentCoordinatesRAS[1]), 2) +
    #                                        math.pow((currentPoint[2] - currentCoordinatesRAS[2]), 2))

    #     targetPoints.append(currentPoint)
    #     distancesToSeed.append(currentDistanceToSeed)

      
    #   # now find the sourcepoint
    #   sourcePointIndex = distancesToSeed.index(min(distancesToSeed))
    #   # .. and remove it after saving it as the sourcePoint
    #   sourcePoint = targetPoints[sourcePointIndex]
    #   distancesToSeed.pop(sourcePointIndex)
    #   targetPoints.pop(sourcePointIndex)

    #   # again, at this point we have a) the sourcePoint and b) a list of real targetPoints
    #   # now create the sourceIdList and targetIdList for the actual centerline computation
    #   sourceIdList = vtk.vtkIdList()
    #   targetIdList = vtk.vtkIdList()

    #   pointLocator = vtk.vtkPointLocator()
    #   pointLocator.SetDataSet(preparedModel)
    #   pointLocator.BuildLocator()

    #   # locate the source on the surface
    #   sourceId = pointLocator.FindClosestPoint(sourcePoint)
    #   sourceIdList.InsertNextId(sourceId)

    #   currentEndPointsMarkupsNode.GetDisplayNode().SetTextScale(0)
    #   currentEndPointsMarkupsNode.RemoveAllMarkups()
    #   currentEndPointsMarkupsNode.AddFiducialFromArray(sourcePoint)

    #   # Calculate centroid of the model 
    #   from vtk.util.numpy_support import vtk_to_numpy
    #   centroid = np.average(vtk_to_numpy(preparedModel.GetPoints().GetData()), axis=0)

    #   # locate the endpoints on the surface
    #   for p in targetPoints:
    #     # fid = currentEndPointsMarkupsNode.AddFiducialFromArray(p)
    #     # currentEndPointsMarkupsNode.SetNthFiducialSelected(fid,False)
    #     # id = pointLocator.FindClosestPoint(p)
    #     # targetIdList.InsertNextId(id)

    #     # Calculate vector of each endpoint to the centroid
    #     pointVector = centroid - p
    #     unitVector = pointVector / np.linalg.norm(pointVector)
    #     pNew = p + unitVector

    #     fid = currentEndPointsMarkupsNode.AddFiducialFromArray(pNew)
    #     currentEndPointsMarkupsNode.SetNthFiducialSelected(fid,False)
    #     id = pointLocator.FindClosestPoint(pNew)
    #     targetIdList.InsertNextId(id)

    #   pathfinding_tupel = self.logic.computeCenterlines(preparedModel, sourceIdList, targetIdList)
    #   pathfinding_network.DeepCopy(pathfinding_tupel[0])
    #   pathfinding_voronoi.DeepCopy(pathfinding_tupel[1])

    #   pathfinding_pts = vtk_to_numpy(pathfinding_network.GetPoints().GetData())


    if not preview:
      # here we start the actual centerline computation which is mathematically more robust and accurate but takes longer than the network extraction

      # clip surface at endpoints identified by the network extraction
      tupel = self.logic.clipSurfaceAtEndPoints(network, currentModelNode.GetPolyData())
      clippedSurface = tupel[0]
      endpoints = tupel[1]

      if pathfindingMode:
        # Write over endpoints array with ONLY the starting point and the indicated ROI point
        # Grab the coordinates of the ROI
        currentCoordinatesROI = [0, 0, 0]
        currentRoiNode.GetNthFiducialPosition(0,currentCoordinatesROI)
        print (currentCoordinatesROI)
        endpoints = vtk.vtkPoints()
        endpoints.InsertPoint(0, currentCoordinatesRAS) # Seed point
        endpoints.InsertPoint(1, currentCoordinatesROI) # Endpoint (ROIs)
        print(endpoints)

      # now find the one endpoint which is closest to the seed and use it as the source point for centerline computation
      # all other endpoints are the target points
      sourcePoint = [0, 0, 0]
      distancesToSeed = []
      targetPoints = []

      # we now need to loop through the endpoints two times
      # first loop is to detect the endpoint resulting in the tiny hole we poked in the surface
      # this is very close to our seed but not the correct sourcePoint
      for i in range(endpoints.GetNumberOfPoints()):
        currentPoint = endpoints.GetPoint(i)
        # get the euclidean distance
        currentDistanceToSeed = math.sqrt(math.pow((currentPoint[0] - currentCoordinatesRAS[0]), 2) +
                                           math.pow((currentPoint[1] - currentCoordinatesRAS[1]), 2) +
                                           math.pow((currentPoint[2] - currentCoordinatesRAS[2]), 2))

        targetPoints.append(currentPoint)
        distancesToSeed.append(currentDistanceToSeed)

      # now we have a list of distances with the corresponding points
      # the index with the most minimal distance is the holePoint, we want to ignore it
      # the index with the second minimal distance is the point closest to the seed, we want to set it as sourcepoint
      # all other points are the targetpoints
      
   #    # get the index of the holePoint, which we want to remove from our endPoints
   #    holePointIndex = distancesToSeed.index(min(distancesToSeed))
   #    # .. and remove it
      
      # # Sometimes not skipping the first point is neccessary for the start point to be correct
   #    if self.removeFirstPointCheckbox.isChecked():
   #      distancesToSeed.pop(holePointIndex)
   #      targetPoints.pop(holePointIndex)
      
      # now find the sourcepoint
      sourcePointIndex = distancesToSeed.index(min(distancesToSeed))
      # .. and remove it after saving it as the sourcePoint
      sourcePoint = targetPoints[sourcePointIndex]
      distancesToSeed.pop(sourcePointIndex)
      targetPoints.pop(sourcePointIndex)

      # again, at this point we have a) the sourcePoint and b) a list of real targetPoints
      # now create the sourceIdList and targetIdList for the actual centerline computation
      sourceIdList = vtk.vtkIdList()
      targetIdList = vtk.vtkIdList()
      print("SourceIDList: ", sourceIdList)
      print("TargetIDList: ", targetIdList)

      pointLocator = vtk.vtkPointLocator()
      pointLocator.SetDataSet(preparedModel)
      pointLocator.BuildLocator()

      # locate the source on the surface
      sourceId = pointLocator.FindClosestPoint(sourcePoint)
      sourceIdList.InsertNextId(sourceId)

      currentEndPointsMarkupsNode.GetDisplayNode().SetTextScale(0)
      currentEndPointsMarkupsNode.RemoveAllMarkups()
      currentEndPointsMarkupsNode.AddFiducialFromArray(sourcePoint)

      # Calculate centroid of the model 
      from vtk.util.numpy_support import vtk_to_numpy
      centroid = np.average(vtk_to_numpy(preparedModel.GetPoints().GetData()), axis=0)

      # locate the endpoints on the surface
      for p in targetPoints:
        # fid = currentEndPointsMarkupsNode.AddFiducialFromArray(p)
        # currentEndPointsMarkupsNode.SetNthFiducialSelected(fid,False)
        # id = pointLocator.FindClosestPoint(p)
        # targetIdList.InsertNextId(id)

        # Calculate vector of each endpoint to the centroid
        pointVector = centroid - p
        unitVector = pointVector / np.linalg.norm(pointVector)
        pNew = p + unitVector

        fid = currentEndPointsMarkupsNode.AddFiducialFromArray(pNew)
        currentEndPointsMarkupsNode.SetNthFiducialSelected(fid,False)
        id = pointLocator.FindClosestPoint(pNew)
        targetIdList.InsertNextId(id)

      tupel = self.logic.computeCenterlines(preparedModel, sourceIdList, targetIdList)
      print (tupel)
      network.DeepCopy(tupel[0])
      print (network)
      voronoi.DeepCopy(tupel[1])
      
      # NEW - debugging
      allPts = vtk_to_numpy(network.GetPoints().GetData())
      print("allPts: ", allPts)
      print("num allPts: ", len(allPts))

      # test findClosestPointOnCenterline function:
      test_point = np.array([0,0,0])
      closestpt = self.findClosestPointOnCenterline(test_point, allPts)
      print("THE CLOSEST PT IS: ", closestpt)

      # Get the list of radius for all points
      # The list is a concatenation of all cell points
      j = 0
      point_data = network.GetPointData()
      radius_array = point_data.GetArray(0)

      # Convert point_data to numpy array
      network_array = vtk_to_numpy(network.GetPoints().GetData())

      # Generate local curvature array
      localcurvature_array = vtk.vtkDoubleArray()
      localcurvature_array.SetName("Local Curvature")
      localcurvature_array.SetNumberOfValues(radius_array.GetMaxId())
      print ("radius array Max id: ", radius_array.GetMaxId())

      # Generate global relative angle array
      globalrelativeangle_array = vtk.vtkDoubleArray()
      globalrelativeangle_array.SetName("GlobalRelativeAngle")
      globalrelativeangle_array.SetNumberOfValues(radius_array.GetMaxId())

      # Generate plane rotation array
      planerotation_array = vtk.vtkDoubleArray()
      planerotation_array.SetName("PlaneRotation")
      planerotation_array.SetNumberOfValues(radius_array.GetMaxId())

      # Generate curvature rate array
      curvaturerate_array = vtk.vtkDoubleArray()
      curvaturerate_array.SetName("Curvature Rate")
      curvaturerate_array.SetNumberOfValues(radius_array.GetMaxId())

      # Calculate trachea reference vector (for GlobalRelativeAngle and PlaneRotation calculations)
      trachea_cell = network.GetCell(0)
      trachea_cell_ids = trachea_cell.GetPointIds()
      trachea_cell_points = trachea_cell.GetPoints()
      num_pts = trachea_cell.GetNumberOfPoints()
      print("num_pts in trachea_cell: ", num_pts)
      trachea_start = np.array([float(s) for s in (str( trachea_cell_points.GetPoint(400) ).replace(' ','')[1:-1]).split(',')])
      # trachea_end = np.array([float(s) for s in (str( trachea_cell_points.GetPoint(num_pts-20) ).replace(' ','')[1:-1]).split(',')])
      trachea_end = np.array([float(s) for s in (str( trachea_cell_points.GetPoint(500) ).replace(' ','')[1:-1]).split(',')])
      trachea_vector = (trachea_end - trachea_start)/np.linalg.norm(trachea_end - trachea_start)
      print ("trachea_vector: ", trachea_vector)

      # Determine the initial reference vector for PlaneRotation calculations
      trachea_cell_ids = trachea_cell.GetPointIds()
      trachea_cell_points = trachea_cell.GetPoints()
      num_pts = trachea_cell.GetNumberOfPoints()
      reference_start = np.array([float(s) for s in (str( trachea_cell_points.GetPoint(600) ).replace(' ','')[1:-1]).split(',')])
      reference_end = np.array([float(s) for s in (str( trachea_cell_points.GetPoint(700) ).replace(' ','')[1:-1]).split(',')])
      reference_vector = (reference_end - reference_start)/np.linalg.norm(reference_end - reference_start)
      print("reference vector: ", reference_vector)
      # before: 600-800

      newPlane = True

      # Track min and max values of each metric
      min_radius = float("inf")
      max_radius = 0.0
      min_localcurv = float("inf")
      max_localcurv = 0.0
      min_globalangle = float("inf")
      max_globalangle = 0.0
      min_curvrate = float("inf")
      max_curvrate = 0.0
      min_planerotation = float("inf")
      max_planerotation = 0.0

      for i in range(radius_array.GetMaxId()):
        curr = radius_array.GetValue(i)
        if curr > max_radius: max_radius = curr
        elif curr < min_radius: min_radius = curr
      print ("Min radius: ", min_radius)
      print ("Max radius: ", max_radius)

      # For curvature rate calculations
      highest = 0.0
      curvature_rate = 0.0
      saved_curvature_rate = 0.0

      # For plane rotation calculations
      threshold_pass_count = 0

      # To remove points with radius smaller than the bronchscope radius
      # bronchoscope_radius = 0.3
      bronchoscope_radius = 0.0

      pointIds = vtk.vtkIdList()
      network.GetLines().GetCell(0, pointIds)
      print(pointIds)

      for i in range( network.GetNumberOfCells() ):
        # Iterate through each cell
        cell = network.GetCell(i)
        cell_ids = cell.GetPointIds()
        cell_points = cell.GetPoints()
    
        if cell.GetNumberOfPoints() > 100:
          for j in range( cell.GetNumberOfPoints() ):
            pt_id = cell_ids.GetId(j)
            pt_r = radius_array.GetValue( pt_id )
            pt_coordinates = network.GetPoint( pt_id )

            if self.colorByLocalCurvatureCheckbox.isChecked() or self.colorByTotalIndexCheckbox.isChecked() or self.colorByCumulativeIndexCheckbox.isChecked():
              # Calculate curvature here
              # if j >= 50 and j < (cell.GetNumberOfPoints() - 50):
              if pt_id >= 50 and pt_id < (localcurvature_array.GetMaxId()-50):
                curr_pt = np.array([float(s) for s in (str( cell_points.GetPoint(j) ).replace(' ','')[1:-1]).split(',')])
                prev_pt = np.array([float(s) for s in (str( cell_points.GetPoint(j-50) ).replace(' ','')[1:-1]).split(',')])
                next_pt = np.array([float(s) for s in (str( cell_points.GetPoint(j+50) ).replace(' ','')[1:-1]).split(',')])

                # Triangle Lengths
                a = np.linalg.norm(next_pt - prev_pt) #a = c-b
                b = np.linalg.norm(next_pt - curr_pt) #b = c-a
                c = np.linalg.norm(prev_pt - curr_pt) #c = b-a
                s = (a + b + c)/2.0

                R = a*b*c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
                local_curvature = 1.0/R * 2550

                # Track min and max values
                if local_curvature > max_localcurv and not np.isnan(local_curvature) and not local_curvature > 10000: max_localcurv = local_curvature
                elif local_curvature < min_localcurv and not np.isnan(local_curvature) and not local_curvature < -10000: min_localcurv = local_curvature

                #if localcurvature_array.GetValue(pt_id) == 0.0:
                if pt_id <= localcurvature_array.GetMaxId() and not np.isnan(local_curvature):
                  localcurvature_array.SetValue(pt_id, local_curvature)

              if 0 <= pt_id <= 51 or (localcurvature_array.GetMaxId()-50) <= pt_id <= localcurvature_array.GetMaxId():
                localcurvature_array.SetValue(pt_id, 0.0)

                #else:
                  #localcurvature_array.SetValue(pt_id, 0.0)

                # to fix value errors for the first and last 30 pts
                # if 0 <= j < 50:
                #   localcurvature_array.SetValue(pt_id, 0.0)
                # if (cell.GetNumberOfPoints()-50) <= j < cell.GetNumberOfPoints():
                #   localcurvature_array.SetValue(pt_id, 0.0)

            if self.colorByGlobalRelativeAngleCheckbox.isChecked() or self.colorByTotalIndexCheckbox.isChecked() or self.colorByCumulativeIndexCheckbox.isChecked():
              # Calculate global relative angle here 
              if j > 0 and j < cell.GetNumberOfPoints():
                # Find the vector of that point to the 10th point ahead of it
                if j < (cell.GetNumberOfPoints() - 10):
                  curr_pt = np.array([float(s) for s in (str( cell_points.GetPoint(j) ).replace(' ','')[1:-1]).split(',')])
                  next_pt = np.array([float(s) for s in (str( cell_points.GetPoint(j+10) ).replace(' ','')[1:-1]).split(',')])
                  direction = (next_pt - curr_pt)/np.linalg.norm(next_pt - curr_pt)

                # Determine similarity to trachea_vector via dot product:
                globalrelativeangle_similiarity = np.dot(direction, trachea_vector)

                # Track min and max values
                if globalrelativeangle_similiarity > max_globalangle: max_globalangle = globalrelativeangle_similiarity
                elif globalrelativeangle_similiarity < min_globalangle: min_globalangle = globalrelativeangle_similiarity

                # Add similiarity metric to global relative angle array
                #if globalrelativeangle_array.GetValue(pt_id) == 0.0:
                if pt_id <= globalrelativeangle_array.GetMaxId():
                    if not np.isnan(globalrelativeangle_similiarity):
                        globalrelativeangle_array.SetValue(pt_id, globalrelativeangle_similiarity)
                    else: 
                        globalrelativeangle_array.SetValue(pt_id, 0.0)
                  #print (globalrelativeangle_similiarity)


            if self.colorByPlaneRotationCheckbox.isChecked() or self.colorByTotalIndexCheckbox.isChecked() or self.colorByCumulativeIndexCheckbox.isChecked():
              
              # new new new
              # TODO

              # Solve for the current vector (10 points ahead of the current point)
              if j > 0 and j < (cell.GetNumberOfPoints() - 10):
                # Find the vector of that point to the 10th point ahead of it
                curr_pt = np.array([float(s) for s in (str( cell_points.GetPoint(j) ).replace(' ','')[1:-1]).split(',')])
                next_pt = np.array([float(s) for s in (str( cell_points.GetPoint(j+10) ).replace(' ','')[1:-1]).split(',')])
                current_vector = (next_pt - curr_pt)/np.linalg.norm(next_pt - curr_pt)

                # Take the cross product of the two reference vectors (trachea vector & current vector) to identify the vector that is normal to the plane defined by the reference vectors
                # The cross product of a and b is a vector perpendicular to both a and b. 
                normal_vector = np.cross(trachea_vector, reference_vector)

                # Solve for the angle between the plane and the current vector
                planerotation_angle = np.arcsin((np.dot(normal_vector, current_vector))/(np.linalg.norm(normal_vector)*np.linalg.norm(current_vector)))
                print("Plane rotation angle: ", planerotation_angle)

                # Add the plane to the scene as a model if the plane changes
                # if newPlane:
                #   planeSource = vtk.vtkPlaneSource()
                #   planeSource.SetOrigin()
                #   planeSource.SetPoint1()
                #   planeSource.SetPoint2()
                #   planeModel = slicer.modules.models.logic().AddModel(planeSource.getOutputPort)
                #   modelDisplay = planeModel.GetDisplayNode()
                #   modelDisplay.SetColor(1,1,0)
                #   modelDisplay.SetBackfaceCulling(0)

                # If the angle between the plane and the current vector reaches a threshold (+/- 0.25?), increment the threshold counter
                if planerotation_angle >= 0.25 or planerotation_angle <= -0.25:
                  threshold_pass_count += 1
                  newPlane = False
                  # Once the threshold has been passed 100 times in a row, update reference vector
                  if threshold_pass_count >= 150:
                    reference_vector = current_vector
                    newPlane = True
                else:
                  threshold_pass_count = 0
                  newPlane = False

                # Track min and max values
                if planerotation_angle > max_planerotation and not np.isnan(planerotation_angle): max_planerotation = planerotation_angle
                elif planerotation_angle < min_planerotation and not np.isnan(planerotation_angle): min_planerotation = planerotation_angle

                # Add plane rotation value to the array
                if pt_id <= planerotation_array.GetMaxId():
                  if not np.isnan(planerotation_angle):
                    planerotation_array.SetValue(pt_id, np.abs(planerotation_angle))
                  else:
                    planerotation_array.SetValue(pt_id, 0.0)           


            if self.colorByCurvatureRateCheckbox.isChecked() or self.colorByTotalIndexCheckbox.isChecked() or self.colorByCumulativeIndexCheckbox.isChecked():
              # Calculate rate of curvature, or overall angle change within a certain amount of distance

              #if j >= 200 and j < (cell.GetNumberOfPoints() - 200):
              if pt_id >= 200 and pt_id < (curvaturerate_array.GetMaxId()-200):
                curr_pt = np.array([float(s) for s in (str( cell_points.GetPoint(j) ).replace(' ','')[1:-1]).split(',')])
                prev_pt = np.array([float(s) for s in (str( cell_points.GetPoint(j-150) ).replace(' ','')[1:-1]).split(',')])
                next_pt = np.array([float(s) for s in (str( cell_points.GetPoint(j+150) ).replace(' ','')[1:-1]).split(',')])

                # Triangle Lengths
                a = np.linalg.norm(next_pt - prev_pt) #a = c-b
                b = np.linalg.norm(next_pt - curr_pt) #b = c-a
                c = np.linalg.norm(prev_pt - curr_pt) #c = b-a
                s = (a + b + c)/2.0

                R = a*b*c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
                curvature_rate = 1.0/R * 2550

                #if j == (cell.GetNumberOfPoints() - 199):
                if pt_id <= (curvaturerate_array.GetMaxId()-200):
                  saved_curvature_rate = curvature_rate
                  print ("saved_curvature_rate: ", saved_curvature_rate)

                #if pt_id <= globalrelativeangle_array.GetMaxId() and not np.isnan(curvature_rate):
                if pt_id <= curvaturerate_array.GetMaxId() and not np.isnan(curvature_rate):
                  curvaturerate_array.SetValue(pt_id, curvature_rate)

                # Track min and max values
                if curvature_rate > max_curvrate: max_curvrate = curvature_rate
                elif curvature_rate < min_curvrate: min_curvrate = curvature_rate

              #if (cell.GetNumberOfPoints() - 200)<j<(cell.GetNumberOfPoints()):
              #elif (curvaturerate_array.GetMaxId()-200)<pt_id<curvaturerate_array.GetMaxId():
              elif pt_id < curvaturerate_array.GetMaxId():
                curvaturerate_array.SetValue(pt_id, saved_curvature_rate)

      if self.colorByTotalIndexCheckbox.isChecked() or self.colorByCumulativeIndexCheckbox.isChecked():

        # Scale all vtk centerline arrays to 0 to 1 by converting to a numpy array
        # Normalization/Interpolation step 
        # radius_array = np.interp(vtk_to_numpy(radius_array), (min_radius, max_radius), (0,1))
        # localcurvature_array = np.interp(vtk_to_numpy(localcurvature_array), (min_localcurv, max_localcurv), (0,1))
        # globalrelativeangle_array = np.interp(vtk_to_numpy(globalrelativeangle_array), (min_globalangle, max_globalangle), (0,1))
        # curvaturerate_array = np.interp(vtk_to_numpy(curvaturerate_array), (min_curvrate, max_curvrate), (0,1))

        # NO INTERPOLATION, just conversion to numpy
        radius_array = vtk_to_numpy(radius_array)
        localcurvature_array = vtk_to_numpy(localcurvature_array)
        globalrelativeangle_array = vtk_to_numpy(globalrelativeangle_array)
        curvaturerate_array = vtk_to_numpy(curvaturerate_array)

        # Remove excess points from the arrays that would prevent broadcasting them together
        minLength = np.min([len(radius_array), len(localcurvature_array), len(globalrelativeangle_array), len(curvaturerate_array)])
        print(minLength)
        radius_array = radius_array[:minLength]
        localcurvature_array = localcurvature_array[:minLength]
        globalrelativeangle_array = globalrelativeangle_array[:minLength]
        curvaturerate_array = curvaturerate_array[:minLength]

        totalindex_array = []
        for i in range(minLength):
          # Tuning of curvature parameters occurs here -- modify the scalar multiplier to affect the weight of each parameter
          # totalindex_array.append(1.0*(100-radius_array[i]) + 1.0*localcurvature_array[i] + 1.0*(100-globalrelativeangle_array[i]) + 1.0*curvaturerate_array[i])
          # totalindex_array.append(radiusScalar*(1-radius_array[i]) + localCurvatureScalar*localcurvature_array[i] + globalAngleScalar*(1-globalrelativeangle_array[i]) + curvatureRateScalar*curvaturerate_array[i])
          # totalindex_array.append(0.5*radiusScalar*(10-radius_array[i]) + 0.02*localCurvatureScalar*localcurvature_array[i] + 2.5*globalAngleScalar*(1-globalrelativeangle_array[i]) + 0.04*curvatureRateScalar*curvaturerate_array[i])
          totalindex_array.append(0.25*(11-radius_array[i]) + 0.06*(11-radius_array[i])*localcurvature_array[i] + 2.5*(1-globalrelativeangle_array[i]) + 0.04*curvaturerate_array[i])


        # Convert all numpy arrays back to vtkdoublearrays
        from vtk.util.numpy_support import numpy_to_vtk
        radius_array = numpy_to_vtk(radius_array)
        localcurvature_array = numpy_to_vtk(localcurvature_array)
        globalrelativeangle_array = numpy_to_vtk(globalrelativeangle_array)
        curvaturerate_array = numpy_to_vtk(curvaturerate_array)
        totalindex_array = numpy_to_vtk(totalindex_array)

        # for k in range(totalindex_array.GetMaxId()):
        #   print (totalindex_array.GetValue(k))

        # Generate total index array label
        totalindex_array.SetName("Total Difficulty Index")


      if self.colorByCumulativeIndexCheckbox.isChecked():
        # Define cumulative index vtkdoublearray
        cumulativeindex_array = vtk.vtkDoubleArray()
        cumulativeindex_array.SetName("Cumulative Difficulty Index")
        cumulativeindex_array.SetNumberOfValues(radius_array.GetMaxId())

        for i in range( network.GetNumberOfCells() ):
          # Iterate through each cell
          cell = network.GetCell(i)
          cell_ids = cell.GetPointIds()
          cell_points = cell.GetPoints()
      
          cumulativeIndex = 0.0
          highest = 45.0

          # Define a threshold value that the Total Difficulty Index must meet in order to be included in the Cumulative Difficulty Index
          thresholdForInclusion = 0.0

          if cell.GetNumberOfPoints() > 100:
            for j in range( cell.GetNumberOfPoints() ):
              pt_id = cell_ids.GetId(j)        

              # Option 1: Add every "difficulty" metric down a branch
              if 0 <= pt_id and pt_id < cumulativeindex_array.GetNumberOfValues():
                currentDifficultyIndex = totalindex_array.GetValue(pt_id)
                if currentDifficultyIndex > thresholdForInclusion and not np.isnan(currentDifficultyIndex):
                    cumulativeIndex += currentDifficultyIndex
                cumulativeindex_array.SetValue(pt_id, cumulativeIndex)
                print (cumulativeIndex)

              # Option 2: Save the highest difficulty metric on the branch so far
              # if 0 <= pt_id and pt_id < cumulativeindex_array.GetNumberOfValues():
              #   curr = totalindex_array.GetValue(pt_id)
              #   if curr > highest: highest = curr
              #   cumulativeindex_array.SetValue(pt_id, highest)


      # Set all pt values in all arrays to 0.0 IF the radius is smaller than the radius of the bronchoscope
      for i in range( network.GetNumberOfCells() ):
        # Iterate through each cell
        cell = network.GetCell(i)
        cell_ids = cell.GetPointIds()
        cell_points = cell.GetPoints()
        for j in range( cell.GetNumberOfPoints() ):
          pt_id = cell_ids.GetId(j)

          if 0 <= pt_id and pt_id < radius_array.GetNumberOfValues()-1:
            pt_r = radius_array.GetValue( pt_id )
            if pt_r < bronchoscope_radius:
              radius_array.SetValue(pt_id, 0.0)
              if self.colorByLocalCurvatureCheckbox.isChecked():
                localcurvature_array.SetValue(pt_id, 0.0)
              if self.colorByGlobalRelativeAngleCheckbox.isChecked():
                globalrelativeangle_array.SetValue(pt_id, 0.0)
              if self.colorByPlaneRotationCheckbox.isChecked():
                planerotation_array.SetValue(pt_id, 0.0)
              if self.colorByCurvatureRateCheckbox.isChecked():
                curvaturerate_array.SetValue(pt_id, 0.0)
              if self.colorByTotalIndexCheckbox.isChecked():
                totalindex_array.SetValue(pt_id, 0.0)
              if self.colorByCumulativeIndexCheckbox.isChecked():
                cumulativeindex_array.SetValue(pt_id, 0.0)

            
      if self.colorByRadiusCheckbox.isChecked():
        self.minRadiusTextbox.setText(min_radius)

      if self.colorByLocalCurvatureCheckbox.isChecked():
        network.GetPointData().AddArray(localcurvature_array)
        self.maxLocalCurvTextbox.setText(max_localcurv)

      if self.colorByGlobalRelativeAngleCheckbox.isChecked():
        network.GetPointData().AddArray(globalrelativeangle_array)
        self.minAngleTextbox.setText(min_globalangle)

      if self.colorByPlaneRotationCheckbox.isChecked():
        network.GetPointData().AddArray(planerotation_array)
        self.maxPlaneRotationTextbox.setText(max_planerotation)
        self.minPlaneRotationTextbox.setText(min_planerotation)


      if self.colorByCurvatureRateCheckbox.isChecked():
        network.GetPointData().AddArray(curvaturerate_array)
        self.maxCurvRateTextbox.setText(round(np.amax(curvaturerate_array),2))

      if self.colorByTotalIndexCheckbox.isChecked():
        network.GetPointData().AddArray(totalindex_array)
        self.maxTotalDifficultyIndexTextbox.setText(round(np.amax(totalindex_array),2))

        self.minRadiusTextbox.setText(min_radius)
        self.maxLocalCurvTextbox.setText(max_localcurv)
        self.minAngleTextbox.setText(min_globalangle)
        self.maxPlaneRotationTextbox.setText(max_planerotation)
        self.minPlaneRotationTextbox.setText(min_planerotation)
        self.maxCurvRateTextbox.setText(max_curvrate)

      if self.colorByCumulativeIndexCheckbox.isChecked():
        network.GetPointData().AddArray(cumulativeindex_array)
        self.maxCumulativeIndexTextbox.setText(round(np.amax(cumulativeindex_array),2))
        self.maxTotalDifficultyIndexTextbox.setText(round(np.amax(totalindex_array),2))

        self.minRadiusTextbox.setText(min_radius)
        self.maxLocalCurvTextbox.setText(max_localcurv)
        self.minAngleTextbox.setText(min_globalangle)
        self.maxPlaneRotationTextbox.setText(max_planerotation)
        self.minPlaneRotationTextbox.setText(min_planerotation)
        self.maxCurvRateTextbox.setText(max_curvrate)
  
      # print("Network: ", network.GetPointData())
      slicer.network = network


      # Write the points for each path to a file
      # ORIGINAL:
      # if self.outputDirectory != '':
      #   for i in range( network.GetNumberOfCells() ):
      #     # Iterate through each cell
      #     cell = network.GetCell(i)
      #     cell_ids = cell.GetPointIds()
      #     cell_points = cell.GetPoints()

      #     if cell.GetNumberOfPoints() > 100:
      #       dir = self.outputDirectory + "\\Branch " + str(i)
      #       if not os.path.exists(dir):
      #         os.makedirs(dir)
      #       with open(dir + "\\raw_data.txt","w") as f:
          
      #         for j in range( cell.GetNumberOfPoints() ):
      #           pt_str = str( cell_points.GetPoint(j) ).replace(' ','')[1:-1]
      #           pt_id = cell_ids.GetId(j)
      #           if 0 <= pt_id and pt_id < radius_array.GetNumberOfValues()-1:
      #             pt_r = radius_array.GetValue( pt_id )
      #             f.write( pt_str + ',' + str( pt_r ) + '\n' )

      # MODIFIED FOR MASA:
      from sys import platform

      if self.outputDirectory != '':
        for i in range( network.GetNumberOfCells() ):
          # Iterate through each cell
          cell = network.GetCell(i)
          cell_ids = cell.GetPointIds()
          cell_points = cell.GetPoints()

          if cell.GetNumberOfPoints() > 100:
            dir = self.outputDirectory

            # if the OS is Windows
            if platform == 'win32':
                if outputFilename == '':
                  outputFilename = "\\raw_data.txt"
                else:
                  outputFilename = "\\" + outputFilename + ".txt"
            # else if the OS is Linux or Mac
            else:
                if outputFilename == '':
                  outputFilename = "/raw_data.txt"
                else:
                  outputFilename = "/" + outputFilename + ".txt"

            if not os.path.exists(dir):
              os.makedirs(dir)
            with open(dir + outputFilename,"w") as f:
          
              for j in range( cell.GetNumberOfPoints() ):
                pt_str = str( cell_points.GetPoint(j) ).replace(' ','')[1:-1]
                pt_id = cell_ids.GetId(j)
                if 0 <= pt_id and pt_id < radius_array.GetNumberOfValues()-1:
                  pt_write = radius_array.GetValue( pt_id )

                  if self.colorByLocalCurvatureCheckbox.isChecked():
                    pt_write = localcurvature_array.GetValue(pt_id)
                  if self.colorByGlobalRelativeAngleCheckbox.isChecked():
                    pt_write = globalrelativeangle_array.GetValue(pt_id)
                  if self.colorByPlaneRotationCheckbox.isChecked():
                    pt_write = planerotation_array.GetValue(pt_id)
                  if self.colorByCurvatureRateCheckbox.isChecked():
                    pt_write = curvaturerate_array.GetValue(pt_id)
                  if self.colorByTotalIndexCheckbox.isChecked():
                    #pt_write = totalindex_array.GetValue(pt_id)
                    pt_1 = radius_array.GetValue(pt_id)
                    pt_2 = localcurvature_array.GetValue(pt_id)
                    pt_3 = globalrelativeangle_array.GetValue(pt_id)
                    pt_4 = curvaturerate_array.GetValue(pt_id)
                    pt_5 = totalindex_array.GetValue(pt_id)
                    pt_write = str(pt_1) + ',' + str(pt_2) + ',' + str(pt_3) + ',' + str(pt_4) + ',' + str(pt_5)
                  if self.colorByCumulativeIndexCheckbox.isChecked():
                    #pt_write = cumulativeindex_array.GetValue(pt_id)
                    pt_1 = radius_array.GetValue(pt_id)
                    pt_2 = localcurvature_array.GetValue(pt_id)
                    pt_3 = globalrelativeangle_array.GetValue(pt_id)
                    pt_4 = curvaturerate_array.GetValue(pt_id)
                    pt_5 = totalindex_array.GetValue(pt_id)
                    pt_6 = cumulativeindex_array.GetValue(pt_id)
                    pt_write = str(pt_1) + ',' + str(pt_2) + ',' + str(pt_3) + ',' + str(pt_4) + ',' + str(pt_5) + ',' + str(pt_6)

                  f.write( str(pt_id) + ',' + pt_str + ',' + str( pt_write ) + '\n' )


    # if pathfindingMode:
    #   centerlineModel = slicer.util.getNode('currentOutputModelNode')
    #   centerlinePoly = centerlineModel.GetPolyData()

    #   for i in range (centerlinePoly.GetNumberOfPoints()):
    #     currentPoint = centerlinePoly.GetPoints().GetPoint(i)
    #     if currentPoint == currentCoordinatesROI:
    #       print ("point found!")
    #       print (currentPoint + "   ")
    #       print (currentCoordinatesROI)
    #       currentId = centerlinePoly.GetPoints().GetId(i)
    #       print (currentId)



    if self.colorByRadiusCheckbox.isChecked():
      # https://gist.github.com/ungi/c1c448fa51cc458d3da75f5e5c73c74c
      slicer.mrmlScene.AddNode(currentOutputModelNode)
      currentOutputModelNode.SetName('OuputModelNode')
      currentOutputModelNode.SetAndObservePolyData(network)
      display = slicer.vtkMRMLModelDisplayNode()
      slicer.mrmlScene.AddNode( display )
      display.SetLineWidth(6)
      currentOutputModelNode.SetAndObserveDisplayNodeID( display.GetID() )
      display.SetActiveScalarName('Radius')
      display.SetAndObserveColorNodeID('vtkMRMLColorTableNodeFileColdToHotRainbow.txt')
      display.SetScalarVisibility(True)

      # if not preview: 
      #   # Print the max and min values of the radius
      #   view=slicer.app.layoutManager().threeDWidget(0).threeDView()
      #   view.cornerAnnotation().SetText(vtk.vtkCornerAnnotation.UpperRight,"Max value: " + str(max_radius))
      #   view.cornerAnnotation().SetText(vtk.vtkCornerAnnotation.UpperLeft,"Min value: " + str(min_radius))
      #   view.cornerAnnotation().GetTextProperty().SetColor(0,0,0)
      #   view.forceRender()

      #### CURRENT BEST SHOT for colortable
      # https://github.com/Slicer/Slicer/blob/master/Modules/Loadable/Colors/Testing/Python/ColorsScalarBarSelfTest.py
      
      # colorWidget = slicer.modules.colors.widgetRepresentation()
      # ctkScalarBarWidget = slicer.util.findChildren(colorWidget, name='VTKScalarBar')[0]
      # ctkScalarBarWidget.setDisplay(1)
      
      # activeColorNodeSelector = slicer.util.findChildren(colorWidget, 'ColorTableComboBox')[0]
      # activeColorNodeSelector.setCurrentNodeID('vtkMRMLColorTableNodeFileColdToHotRainbow.txt')
      ########

    elif self.colorByLocalCurvatureCheckbox.isChecked():
      # https://gist.github.com/ungi/c1c448fa51cc458d3da75f5e5c73c74c
      slicer.mrmlScene.AddNode(currentOutputModelNode)
      currentOutputModelNode.SetName('OuputModelNode')
      currentOutputModelNode.SetAndObservePolyData(network)
      display = slicer.vtkMRMLModelDisplayNode()
      slicer.mrmlScene.AddNode( display )
      display.SetLineWidth(6)
      currentOutputModelNode.SetAndObserveDisplayNodeID( display.GetID() )
      display.SetActiveScalarName('Local Curvature')
      display.SetAndObserveColorNodeID('vtkMRMLColorTableNodeFileColdToHotRainbow.txt')
      display.SetScalarVisibility(True)

    elif self.colorByGlobalRelativeAngleCheckbox.isChecked():
      # https://gist.github.com/ungi/c1c448fa51cc458d3da75f5e5c73c74c
      slicer.mrmlScene.AddNode(currentOutputModelNode)
      currentOutputModelNode.SetName('OuputModelNode')
      currentOutputModelNode.SetAndObservePolyData(network)
      display = slicer.vtkMRMLModelDisplayNode()
      slicer.mrmlScene.AddNode( display )
      display.SetLineWidth(6)
      currentOutputModelNode.SetAndObserveDisplayNodeID( display.GetID() )
      display.SetActiveScalarName('GlobalRelativeAngle')
      display.SetAndObserveColorNodeID('vtkMRMLColorTableNodeFileColdToHotRainbow.txt')
      display.SetScalarVisibility(True)

    elif self.colorByPlaneRotationCheckbox.isChecked():
      # https://gist.github.com/ungi/c1c448fa51cc458d3da75f5e5c73c74c
      slicer.mrmlScene.AddNode(currentOutputModelNode)
      currentOutputModelNode.SetName('OuputModelNode')
      currentOutputModelNode.SetAndObservePolyData(network)
      display = slicer.vtkMRMLModelDisplayNode()
      slicer.mrmlScene.AddNode( display )
      display.SetLineWidth(6)
      currentOutputModelNode.SetAndObserveDisplayNodeID( display.GetID() )
      display.SetActiveScalarName('PlaneRotation')
      display.SetAndObserveColorNodeID('vtkMRMLColorTableNodeFileColdToHotRainbow.txt')
      display.SetScalarVisibility(True)

    elif self.colorByCurvatureRateCheckbox.isChecked():
      # https://gist.github.com/ungi/c1c448fa51cc458d3da75f5e5c73c74c
      slicer.mrmlScene.AddNode(currentOutputModelNode)
      currentOutputModelNode.SetName('OuputModelNode')
      currentOutputModelNode.SetAndObservePolyData(network)
      display = slicer.vtkMRMLModelDisplayNode()
      slicer.mrmlScene.AddNode( display )
      display.SetLineWidth(6)
      currentOutputModelNode.SetAndObserveDisplayNodeID( display.GetID() )
      display.SetActiveScalarName('Curvature Rate')
      display.SetAndObserveColorNodeID('vtkMRMLColorTableNodeFileColdToHotRainbow.txt')
      display.SetScalarVisibility(True)

    elif self.colorByTotalIndexCheckbox.isChecked():
      # https://gist.github.com/ungi/c1c448fa51cc458d3da75f5e5c73c74c
      slicer.mrmlScene.AddNode(currentOutputModelNode)
      currentOutputModelNode.SetName('OuputModelNode')
      currentOutputModelNode.SetAndObservePolyData(network)
      display = slicer.vtkMRMLModelDisplayNode()
      slicer.mrmlScene.AddNode( display )
      display.SetLineWidth(6)
      currentOutputModelNode.SetAndObserveDisplayNodeID( display.GetID() )
      display.SetActiveScalarName('Total Difficulty Index')
      display.SetAndObserveColorNodeID('vtkMRMLColorTableNodeFileColdToHotRainbow.txt')
      display.SetScalarVisibility(True)

    elif self.colorByCumulativeIndexCheckbox.isChecked():
      # https://gist.github.com/ungi/c1c448fa51cc458d3da75f5e5c73c74c
      slicer.mrmlScene.AddNode(currentOutputModelNode)
      currentOutputModelNode.SetName('OuputModelNode')
      currentOutputModelNode.SetAndObservePolyData(network)
      display = slicer.vtkMRMLModelDisplayNode()
      slicer.mrmlScene.AddNode( display )
      display.SetLineWidth(6)
      currentOutputModelNode.SetAndObserveDisplayNodeID( display.GetID() )
      display.SetActiveScalarName('Cumulative Difficulty Index')
      display.SetAndObserveColorNodeID('vtkMRMLColorTableNodeFileColdToHotRainbow.txt')
      display.SetScalarVisibility(True)

    else:
      currentOutputModelNode.SetAndObservePolyData(network)



    # Make model node semi-transparent to make centerline inside visible
    currentModelNode.CreateDefaultDisplayNodes()
    currentModelDisplayNode = currentModelNode.GetDisplayNode()
    currentModelDisplayNode.SetOpacity(0.4)

    if currentVoronoiModelNode:
      # Configure the displayNode to show the centerline and Voronoi model
      currentOutputModelNode.CreateDefaultDisplayNodes()
      currentOutputModelDisplayNode = currentOutputModelNode.GetDisplayNode()
      currentOutputModelDisplayNode.SetColor(0.0, 0.0, 0.4)  # red
      currentOutputModelDisplayNode.SetBackfaceCulling(0)
      currentOutputModelDisplayNode.SetSliceIntersectionVisibility(0)
      currentOutputModelDisplayNode.SetVisibility(1)
      currentOutputModelDisplayNode.SetOpacity(1.0)

    # only update the voronoi node if we are not in preview mode

    if currentVoronoiModelNode and not preview:
      currentVoronoiModelNode.SetAndObservePolyData(voronoi)
      currentVoronoiModelNode.CreateDefaultDisplayNodes()
      currentVoronoiModelDisplayNode = currentVoronoiModelNode.GetDisplayNode()

      # always configure the displayNode to show the model
      currentVoronoiModelDisplayNode.SetScalarVisibility(1)
      currentVoronoiModelDisplayNode.SetBackfaceCulling(0)
      currentVoronoiModelDisplayNode.SetActiveScalarName("Radius")
      currentVoronoiModelDisplayNode.SetAndObserveColorNodeID(slicer.mrmlScene.GetNodesByName("Labels").GetItemAsObject(0).GetID())
      currentVoronoiModelDisplayNode.SetSliceIntersectionVisibility(0)
      currentVoronoiModelDisplayNode.SetVisibility(1)
      currentVoronoiModelDisplayNode.SetOpacity(0.5)

    logging.debug("End of Centerline Computation..")

    return True

class CenterlineComputationLogic(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''

    def prepareModel(self, polyData):
        '''
        '''
        # import the vmtk libraries
        try:
            import vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry
            import vtkvmtkMiscPython as vtkvmtkMisc
        except ImportError:
            logging.error("Unable to import the SlicerVmtk libraries")

        capDisplacement = 0.0

        surfaceCleaner = vtk.vtkCleanPolyData()
        surfaceCleaner.SetInputData(polyData)
        surfaceCleaner.Update()

        surfaceTriangulator = vtk.vtkTriangleFilter()
        surfaceTriangulator.SetInputData(surfaceCleaner.GetOutput())
        surfaceTriangulator.PassLinesOff()
        surfaceTriangulator.PassVertsOff()
        surfaceTriangulator.Update()

        # new steps for preparation to avoid problems because of slim models (f.e. at stenosis)
        subdiv = vtk.vtkLinearSubdivisionFilter()
        subdiv.SetInputData(surfaceTriangulator.GetOutput())
        subdiv.SetNumberOfSubdivisions(1)
        subdiv.Update()

        smooth = vtk.vtkWindowedSincPolyDataFilter()
        smooth.SetInputData(subdiv.GetOutput())
        smooth.SetNumberOfIterations(20)
        smooth.SetPassBand(0.1)
        smooth.SetBoundarySmoothing(1)
        smooth.Update()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(smooth.GetOutput())
        normals.SetAutoOrientNormals(1)
        normals.SetFlipNormals(0)
        normals.SetConsistency(1)
        normals.SplittingOff()
        normals.Update()

        surfaceCapper = vtkvmtkComputationalGeometry.vtkvmtkCapPolyData()
        surfaceCapper.SetInputData(normals.GetOutput())
        surfaceCapper.SetDisplacement(capDisplacement)
        surfaceCapper.SetInPlaneDisplacement(capDisplacement)
        surfaceCapper.Update()

        outPolyData = vtk.vtkPolyData()
        outPolyData.DeepCopy(surfaceCapper.GetOutput())

        return outPolyData


    def decimateSurface(self, polyData):
        '''
        '''

        decimationFilter = vtk.vtkDecimatePro()
        decimationFilter.SetInputData(polyData)
        decimationFilter.SetTargetReduction(0.99)
        decimationFilter.SetBoundaryVertexDeletion(0)
        decimationFilter.PreserveTopologyOn()
        decimationFilter.Update()

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(decimationFilter.GetOutput())
        cleaner.Update()

        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputData(cleaner.GetOutput())
        triangleFilter.Update()

        outPolyData = vtk.vtkPolyData()
        outPolyData.DeepCopy(triangleFilter.GetOutput())

        return outPolyData


    def openSurfaceAtPoint(self, polyData, seed):
        '''
        Returns a new surface with an opening at the given seed.
        '''

        someradius = 1.0

        pointLocator = vtk.vtkPointLocator()
        pointLocator.SetDataSet(polyData)
        pointLocator.BuildLocator()

        # find the closest point next to the seed on the surface
        # id = pointLocator.FindClosestPoint(int(seed[0]),int(seed[1]),int(seed[2]))
        id = pointLocator.FindClosestPoint(seed)

        # the seed is now guaranteed on the surface
        seed = polyData.GetPoint(id)

        sphere = vtk.vtkSphere()
        sphere.SetCenter(seed[0], seed[1], seed[2])
        sphere.SetRadius(someradius)

        clip = vtk.vtkClipPolyData()
        clip.SetInputData(polyData)
        clip.SetClipFunction(sphere)
        clip.Update()

        outPolyData = vtk.vtkPolyData()
        outPolyData.DeepCopy(clip.GetOutput())

        return outPolyData



    def extractNetwork(self, polyData):
        '''
        Returns the network of the given surface.
        '''
        # import the vmtk libraries
        try:
            import vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry
            import vtkvmtkMiscPython as vtkvmtkMisc
        except ImportError:
            logging.error("Unable to import the SlicerVmtk libraries")

        radiusArrayName = 'Radius'
        topologyArrayName = 'Topology'
        marksArrayName = 'Marks'

        networkExtraction = vtkvmtkMisc.vtkvmtkPolyDataNetworkExtraction()
        networkExtraction.SetInputData(polyData)
        networkExtraction.SetAdvancementRatio(1.05)
        networkExtraction.SetRadiusArrayName(radiusArrayName)
        networkExtraction.SetTopologyArrayName(topologyArrayName)
        networkExtraction.SetMarksArrayName(marksArrayName)
        networkExtraction.Update()

        outPolyData = vtk.vtkPolyData()
        outPolyData.DeepCopy(networkExtraction.GetOutput())

        return outPolyData


    def clipSurfaceAtEndPoints(self, networkPolyData, surfacePolyData):
        '''
        Clips the surfacePolyData on the endpoints identified using the networkPolyData.

        Returns a tupel of the form [clippedPolyData, endpointsPoints]
        '''
        # import the vmtk libraries
        try:
            import vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry
            import vtkvmtkMiscPython as vtkvmtkMisc
        except ImportError:
            logging.error("Unable to import the SlicerVmtk libraries")

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(networkPolyData)
        cleaner.Update()
        network = cleaner.GetOutput()
        network.BuildCells()
        network.BuildLinks(0)
        endpointIds = vtk.vtkIdList()

        radiusArray = network.GetPointData().GetArray('Radius')

        endpoints = vtk.vtkPolyData()
        endpointsPoints = vtk.vtkPoints()
        endpointsRadius = vtk.vtkDoubleArray()
        endpointsRadius.SetName('Radius')
        endpoints.SetPoints(endpointsPoints)
        endpoints.GetPointData().AddArray(endpointsRadius)

        radiusFactor = 1.2
        minRadius = 0.01

        for i in range(network.GetNumberOfCells()):
            numberOfCellPoints = network.GetCell(i).GetNumberOfPoints()
            pointId0 = network.GetCell(i).GetPointId(0)
            pointId1 = network.GetCell(i).GetPointId(numberOfCellPoints - 1)

            pointCells = vtk.vtkIdList()
            network.GetPointCells(pointId0, pointCells)
            numberOfEndpoints = endpointIds.GetNumberOfIds()
            if pointCells.GetNumberOfIds() == 1:
                pointId = endpointIds.InsertUniqueId(pointId0)
                if pointId == numberOfEndpoints:
                    point = network.GetPoint(pointId0)
                    radius = radiusArray.GetValue(pointId0)
                    radius = max(radius, minRadius)
                    endpointsPoints.InsertNextPoint(point)
                    endpointsRadius.InsertNextValue(radiusFactor * radius)

            pointCells = vtk.vtkIdList()
            network.GetPointCells(pointId1, pointCells)
            numberOfEndpoints = endpointIds.GetNumberOfIds()
            if pointCells.GetNumberOfIds() == 1:
                pointId = endpointIds.InsertUniqueId(pointId1)
                if pointId == numberOfEndpoints:
                    point = network.GetPoint(pointId1)
                    radius = radiusArray.GetValue(pointId1)
                    radius = max(radius, minRadius)
                    endpointsPoints.InsertNextPoint(point)
                    endpointsRadius.InsertNextValue(radiusFactor * radius)

        polyBall = vtkvmtkComputationalGeometry.vtkvmtkPolyBall()
        #polyBall.SetInputData(endpoints)
        polyBall.SetInput(endpoints)
        polyBall.SetPolyBallRadiusArrayName('Radius')

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(surfacePolyData)
        clipper.SetClipFunction(polyBall)
        clipper.Update()

        connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
        connectivityFilter.SetInputData(clipper.GetOutput())
        connectivityFilter.ColorRegionsOff()
        connectivityFilter.SetExtractionModeToLargestRegion()
        connectivityFilter.Update()

        clippedSurface = connectivityFilter.GetOutput()

        outPolyData = vtk.vtkPolyData()
        outPolyData.DeepCopy(clippedSurface)

        return [outPolyData, endpointsPoints]


    def computeCenterlines(self, polyData, inletSeedIds, outletSeedIds):
        '''
        Returns a tupel of two vtkPolyData objects.
        The first are the centerlines, the second is the corresponding Voronoi diagram.
        '''
        # import the vmtk libraries
        try:
            import vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry
            import vtkvmtkMiscPython as vtkvmtkMisc
        except ImportError:
            logging.error("Unable to import the SlicerVmtk libraries")

        flipNormals = 0
        radiusArrayName = 'Radius'
        costFunction = '1/R'


        centerlineFilter = vtkvmtkComputationalGeometry.vtkvmtkPolyDataCenterlines()
        centerlineFilter.SetInputData(polyData)
        centerlineFilter.SetSourceSeedIds(inletSeedIds)
        centerlineFilter.SetTargetSeedIds(outletSeedIds)
        centerlineFilter.SetRadiusArrayName(radiusArrayName)
        centerlineFilter.SetCostFunction(costFunction)
        centerlineFilter.SetFlipNormals(flipNormals)
        centerlineFilter.SetAppendEndPointsToCenterlines(0)
        centerlineFilter.SetSimplifyVoronoi(0)
        centerlineFilter.SetCenterlineResampling(0)
        centerlineFilter.SetResamplingStepLength(1.0)
        centerlineFilter.Update()

        outPolyData = vtk.vtkPolyData()
        outPolyData.DeepCopy(centerlineFilter.GetOutput())

        outPolyData2 = vtk.vtkPolyData()
        outPolyData2.DeepCopy(centerlineFilter.GetVoronoiDiagram())

        return [outPolyData, outPolyData2]



    
class Slicelet(object):
  """A slicer slicelet is a module widget that comes up in stand alone mode
  implemented as a python class.
  This class provides common wrapper functionality used by all slicer modlets.
  """
  # TODO: put this in a SliceletLib
  # TODO: parse command line arge


  def __init__(self, widgetClass=None):
    self.parent = qt.QFrame()
    self.parent.setLayout(qt.QVBoxLayout())

    # TODO: should have way to pop up python interactor
    self.buttons = qt.QFrame()
    self.buttons.setLayout(qt.QHBoxLayout())
    self.parent.layout().addWidget(self.buttons)
    self.addDataButton = qt.QPushButton("Add Data")
    self.buttons.layout().addWidget(self.addDataButton)
    self.addDataButton.connect("clicked()", slicer.app.ioManager().openAddDataDialog)
    self.loadSceneButton = qt.QPushButton("Load Scene")
    self.buttons.layout().addWidget(self.loadSceneButton)
    self.loadSceneButton.connect("clicked()", slicer.app.ioManager().openLoadSceneDialog)

class CenterlineComputationSlicelet(Slicelet):
  """ Creates the interface when module is run as a stand alone gui app.
  """

  def __init__(self):
    super(CenterlineComputationSlicelet, self).__init__(CenterlineComputationWidget)


if __name__ == "__main__":
  # TODO: need a way to access and parse command line arguments
  # TODO: ideally command line args should handle --xml

  import sys
  print(sys.argv)

  slicelet = CenterlineComputationSlicelet()
