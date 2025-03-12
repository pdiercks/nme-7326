# trace generated using paraview version 5.11.0

import sys
import argparse
# import paraview
# paraview.compatibility.major = 5
# paraview.compatibility.minor = 11

from paraview.simple import (
        _DisableFirstRenderCameraReset,
        ColorBy,
        GetActiveViewOrCreate,
        GetAnimationScene,
        GetLayout,
        GetColorTransferFunction,
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

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # show data in view
    fieldspvdDisplay = Show(fieldspvd, renderView1, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    fieldspvdDisplay.Representation = 'Surface'
    fieldspvdDisplay.ColorArrayName = [None, '']
    fieldspvdDisplay.SelectTCoordArray = 'None'
    fieldspvdDisplay.SelectNormalArray = 'None'
    fieldspvdDisplay.SelectTangentArray = 'None'
    fieldspvdDisplay.OSPRayScaleArray = 'u_err'
    fieldspvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    fieldspvdDisplay.SelectOrientationVectors = 'None'
    fieldspvdDisplay.ScaleFactor = 40.000000000000014
    fieldspvdDisplay.SelectScaleArray = 'None'
    fieldspvdDisplay.GlyphType = 'Arrow'
    fieldspvdDisplay.GlyphTableIndexArray = 'None'
    fieldspvdDisplay.GaussianRadius = 2.0000000000000004
    fieldspvdDisplay.SetScaleArray = ['POINTS', 'u_err']
    fieldspvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    fieldspvdDisplay.OpacityArray = ['POINTS', 'u_err']
    fieldspvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    fieldspvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
    fieldspvdDisplay.PolarAxes = 'PolarAxesRepresentation'
    fieldspvdDisplay.ScalarOpacityUnitDistance = 3.6320069810034905
    fieldspvdDisplay.OpacityArrayName = ['POINTS', 'u_err']
    fieldspvdDisplay.SelectInputVectors = ['POINTS', 'u_err']
    fieldspvdDisplay.WriteLog = ''

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    fieldspvdDisplay.ScaleTransferFunction.Points = [-0.0005371206773280637, 0.0, 0.5, 0.0, 0.0008589852223098368, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    fieldspvdDisplay.OpacityTransferFunction.Points = [-0.0005371206773280637, 0.0, 0.5, 0.0, 0.0008589852223098368, 1.0, 0.5, 0.0]

    # reset view to fit data
    renderView1.ResetCamera(False)

    # changing interaction mode based on data extents
    renderView1.CameraPosition = [200.00000000000006, 200.00000000000006, 10000.0]
    renderView1.CameraFocalPoint = [200.00000000000006, 200.00000000000006, 0.0]

    # update the view to ensure updated data information
    renderView1.Update()

    # set scalar coloring
    ColorBy(fieldspvdDisplay, ('FIELD', 'vtkBlockColors'))

    # show color bar/color legend
    fieldspvdDisplay.SetScalarBarVisibility(renderView1, True)

    # get color transfer function/color map for 'vtkBlockColors'
    vtkBlockColorsLUT = GetColorTransferFunction('vtkBlockColors')

    # get opacity transfer function/opacity map for 'vtkBlockColors'
    # vtkBlockColorsPWF = GetOpacityTransferFunction('vtkBlockColors')

    # get 2D transfer function for 'vtkBlockColors'
    # vtkBlockColorsTF2D = GetTransferFunction2D('vtkBlockColors')

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

    # get 2D transfer function for 'u_err'
    # u_errTF2D = GetTransferFunction2D('u_err')

    # create a new 'Warp By Vector'
    warpByVector1 = WarpByVector(registrationName='WarpByVector1', Input=fieldspvd)
    warpByVector1.Vectors = ['POINTS', 'u_err']

    # Properties modified on warpByVector1
    warpByVector1.Vectors = ['POINTS', 'u_rom_local']
    warpByVector1.ScaleFactor = 10.0

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
    warpByVector1Display.ScaleFactor = 45.1698657762098
    warpByVector1Display.SelectScaleArray = 'None'
    warpByVector1Display.GlyphType = 'Arrow'
    warpByVector1Display.GlyphTableIndexArray = 'None'
    warpByVector1Display.GaussianRadius = 2.25849328881049
    warpByVector1Display.SetScaleArray = ['POINTS', 'u_err']
    warpByVector1Display.ScaleTransferFunction = 'PiecewiseFunction'
    warpByVector1Display.OpacityArray = ['POINTS', 'u_err']
    warpByVector1Display.OpacityTransferFunction = 'PiecewiseFunction'
    warpByVector1Display.DataAxesGrid = 'GridAxesRepresentation'
    warpByVector1Display.PolarAxes = 'PolarAxesRepresentation'
    warpByVector1Display.ScalarOpacityFunction = u_errPWF
    warpByVector1Display.ScalarOpacityUnitDistance = 4.013819064600196
    warpByVector1Display.OpacityArrayName = ['POINTS', 'u_err']
    warpByVector1Display.SelectInputVectors = ['POINTS', 'u_err']
    warpByVector1Display.WriteLog = ''

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    warpByVector1Display.ScaleTransferFunction.Points = [-0.0005371206773280637, 0.0, 0.5, 0.0, 0.0008589852223098368, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    warpByVector1Display.OpacityTransferFunction.Points = [-0.0005371206773280637, 0.0, 0.5, 0.0, 0.0008589852223098368, 1.0, 0.5, 0.0]

    # hide data in view
    Hide(fieldspvd, renderView1)

    # show color bar/color legend
    warpByVector1Display.SetScalarBarVisibility(renderView1, True)

    # get color legend/bar for u_errLUT in view renderView1
    color_bar = GetScalarBar(u_errLUT, renderView1)
    color_bar.AutoOrient = 0
    color_bar.Orientation = 'Vertical'
    color_bar.WindowLocation = 'Any Location'
    color_bar.Position = [0.33, 0.12]
    color_bar.ScalarBarLength = 0.33
    color_bar.Title = "$\\vert u_{\\mathrm{fom}}-u_{\\mathrm{rom}}\\vert$"
    color_bar.ComponentTitle = ""
    color_bar.TitleFontFamily = "Times"
    color_bar.TitleFontSize = 30
    color_bar.LabelFontFamily = "Times"
    color_bar.LabelFontSize = int(30 * 0.95)

    u_errLUT.RescaleTransferFunction(0.0, 0.0001)

    # update the view to ensure updated data information
    renderView1.Update()

    # get layout
    layout1 = GetLayout()

    # layout/tab size in pixels
    layout1.SetSize(1538, 789)

    # current camera placement for renderView1
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [420.82523559488646, 215.05626606328775, 10000.0]
    renderView1.CameraFocalPoint = [420.82523559488646, 215.05626606328775, 0.0]
    renderView1.CameraParallelScale = 282.842712474619

    # save screenshot
    SaveScreenshot(png_file, renderView1, ImageResolution=[2000, 1026])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pvdfile", type=str, help="The PVD File to read.")
    parser.add_argument("png", type=str, help="The output png.")
    args = parser.parse_args(sys.argv[1:])
    main(args.pvdfile, args.png)
