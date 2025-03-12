# trace generated using paraview version 5.11.0

import sys
import argparse
from paraview.simple import (
        ColorBy,
        _DisableFirstRenderCameraReset,
        GetActiveViewOrCreate,
        GetAnimationScene,
        GetColorTransferFunction,
        GetLayout,
        GetOpacityTransferFunction,
        GetScalarBar,
        Hide,
        HideScalarBarIfNotNeeded,
        PVDReader,
        SaveScreenshot,
        Show,
        WarpByVector,
        )


def main(pvd_file, png_file):
    _DisableFirstRenderCameraReset()
    fieldspvd = PVDReader(registrationName='fields.pvd', FileName=pvd_file)
    fieldspvd.CellArrays = ['vtkGhostType', 'vtkOriginalCellIds']
    fieldspvd.PointArrays = ['vtkOriginalPointIds', 'vtkGhostType', 'u_fom_local', 'u_rom_local', 'u_err']

    # get animation scene
    animationScene1 = GetAnimationScene()

    # update animation scene based on data timesteps
    animationScene1.UpdateAnimationUsingDataTimeSteps()

    renderView1 = GetActiveViewOrCreate('RenderView')
    fieldspvdDisplay = Show(fieldspvd, renderView1, 'UnstructuredGridRepresentation')

    fieldspvdDisplay.Representation = 'Surface'
    fieldspvdDisplay.ColorArrayName = [None, '']
    fieldspvdDisplay.SelectTCoordArray = 'None'
    fieldspvdDisplay.SelectNormalArray = 'None'
    fieldspvdDisplay.SelectTangentArray = 'None'
    fieldspvdDisplay.OSPRayScaleArray = 'u_err'
    fieldspvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    fieldspvdDisplay.SelectOrientationVectors = 'None'
    fieldspvdDisplay.ScaleFactor = 0.5000000000000001
    fieldspvdDisplay.SelectScaleArray = 'None'
    fieldspvdDisplay.GlyphType = 'Arrow'
    fieldspvdDisplay.GlyphTableIndexArray = 'None'
    fieldspvdDisplay.GaussianRadius = 0.025000000000000005
    fieldspvdDisplay.SetScaleArray = ['POINTS', 'u_err']
    fieldspvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    fieldspvdDisplay.OpacityArray = ['POINTS', 'u_err']
    fieldspvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    fieldspvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
    fieldspvdDisplay.PolarAxes = 'PolarAxesRepresentation'
    fieldspvdDisplay.ScalarOpacityUnitDistance = 0.18373563033147738
    fieldspvdDisplay.OpacityArrayName = ['POINTS', 'u_err']
    fieldspvdDisplay.SelectInputVectors = ['POINTS', 'u_err']
    fieldspvdDisplay.WriteLog = ''

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    fieldspvdDisplay.ScaleTransferFunction.Points = [-0.0001678770020527298, 0.0, 0.5, 0.0, 0.00016706741933814054, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    fieldspvdDisplay.OpacityTransferFunction.Points = [-0.0001678770020527298, 0.0, 0.5, 0.0, 0.00016706741933814054, 1.0, 0.5, 0.0]

    # reset view to fit data
    renderView1.ResetCamera(False)

    # changing interaction mode based on data extents
    renderView1.CameraPosition = [2.5000000000000004, 2.5000000000000004, 10000.0]
    renderView1.CameraFocalPoint = [2.5000000000000004, 2.5000000000000004, 0.0]

    # update the view to ensure updated data information
    renderView1.Update()

    # set scalar coloring
    ColorBy(fieldspvdDisplay, ('FIELD', 'vtkBlockColors'))

    # show color bar/color legend
    fieldspvdDisplay.SetScalarBarVisibility(renderView1, True)

    # get color transfer function/color map for 'vtkBlockColors'
    vtkBlockColorsLUT = GetColorTransferFunction('vtkBlockColors')

    # set scalar coloring
    ColorBy(fieldspvdDisplay, ('POINTS', 'u_err', 'Magnitude'))

    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(vtkBlockColorsLUT, renderView1)

    # rescale color and/or opacity maps used to include current data range
    fieldspvdDisplay.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    fieldspvdDisplay.SetScalarBarVisibility(renderView1, True)

    # get color transfer function/color map for 'u_err'
    u_errLUT = GetColorTransferFunction('u_err')

    # get opacity transfer function/opacity map for 'u_err'
    u_errPWF = GetOpacityTransferFunction('u_err')

    # Rescale transfer function
    u_errLUT.RescaleTransferFunction(0.0, 0.0001)

    # Rescale transfer function
    u_errPWF.RescaleTransferFunction(0.0, 0.0001)

    # create a new 'Warp By Vector'
    warpByVector1 = WarpByVector(registrationName='WarpByVector1', Input=fieldspvd)
    warpByVector1.Vectors = ['POINTS', 'u_err']

    # Properties modified on warpByVector1
    warpByVector1.Vectors = ['POINTS', 'u_rom_local']

    # show data in view
    warpByVector1Display = Show(warpByVector1, renderView1, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    warpByVector1Display.Representation = 'Surface'
    warpByVector1Display.ColorArrayName = ['POINTS', 'u_err']
    warpByVector1Display.LookupTable = u_errLUT
    warpByVector1Display.SelectTCoordArray = 'None'
    warpByVector1Display.SelectNormalArray = 'None'
    warpByVector1Display.SelectTangentArray = 'None'
    warpByVector1Display.OSPRayScaleArray = 'u_err'
    warpByVector1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    warpByVector1Display.SelectOrientationVectors = 'None'
    warpByVector1Display.ScaleFactor = 0.5786012155676312
    warpByVector1Display.SelectScaleArray = 'None'
    warpByVector1Display.GlyphType = 'Arrow'
    warpByVector1Display.GlyphTableIndexArray = 'None'
    warpByVector1Display.GaussianRadius = 0.02893006077838156
    warpByVector1Display.SetScaleArray = ['POINTS', 'u_err']
    warpByVector1Display.ScaleTransferFunction = 'PiecewiseFunction'
    warpByVector1Display.OpacityArray = ['POINTS', 'u_err']
    warpByVector1Display.OpacityTransferFunction = 'PiecewiseFunction'
    warpByVector1Display.DataAxesGrid = 'GridAxesRepresentation'
    warpByVector1Display.PolarAxes = 'PolarAxesRepresentation'
    warpByVector1Display.ScalarOpacityFunction = u_errPWF
    warpByVector1Display.ScalarOpacityUnitDistance = 0.20955889483363713
    warpByVector1Display.OpacityArrayName = ['POINTS', 'u_err']
    warpByVector1Display.SelectInputVectors = ['POINTS', 'u_err']
    warpByVector1Display.WriteLog = ''

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    warpByVector1Display.ScaleTransferFunction.Points = [-0.0001678770020527298, 0.0, 0.5, 0.0, 0.00016706741933814054, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    warpByVector1Display.OpacityTransferFunction.Points = [-0.0001678770020527298, 0.0, 0.5, 0.0, 0.00016706741933814054, 1.0, 0.5, 0.0]

    # hide data in view
    Hide(fieldspvd, renderView1)

    # show color bar/color legend
    warpByVector1Display.SetScalarBarVisibility(renderView1, True)

    # Rescale transfer function
    u_errLUT.RescaleTransferFunction(0.0, 0.0001)

    # Rescale transfer function
    u_errPWF.RescaleTransferFunction(0.0, 0.0001)

    # update the view to ensure updated data information
    renderView1.Update()

    # change scalar bar placement
    color_bar = GetScalarBar(u_errLUT, renderView1)
    color_bar.AutoOrient = 0
    color_bar.Orientation = 'Horizontal'
    color_bar.WindowLocation = 'Any Location'

    # change scalar bar placement
    color_bar.Position = [0.24956217162872152, 0.11441144114411442]
    color_bar.ScalarBarLength = 0.3300000000000003
    color_bar.Title = "$\\vert u_{\\mathrm{fom}}-u_{\\mathrm{rom}}\\vert$"
    color_bar.ComponentTitle = ""
    color_bar.TitleFontFamily = "Times"
    color_bar.TitleFontSize = 30
    color_bar.LabelFontFamily = "Times"
    color_bar.LabelFontSize = int(30 * 0.95)

    # get layout
    layout1 = GetLayout()

    # layout/tab size in pixels
    layout1.SetSize(1142, 909)

    # current camera placement for renderView1
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [3.7330417589205687, 2.255274154718055, 10000.0]
    renderView1.CameraFocalPoint = [3.7330417589205687, 2.255274154718055, 0.0]
    renderView1.CameraParallelScale = 4.277996026178613

    # save screenshot
    SaveScreenshot(png_file, renderView1, ImageResolution=[2000, 1590])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pvdfile", type=str, help="The PVD File to read.")
    parser.add_argument("png", type=str, help="The output png.")
    args = parser.parse_args(sys.argv[1:])
    main(args.pvdfile, args.png)
