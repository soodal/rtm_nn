# rtm_nn

[Perpose]
The simple Neural Network Model for Emulate the Radiative Transfer Model VLIDORT.

[Workflow]

Prepare VLIDORT Look-up-table data.

Read VLIDORT LUT.

Train and validate VLIDORT LUT data.


[Radiative Transfer Model]

VLIDORT

[[Input]]
Solar Zenith Angle
Viewing Zenith Angle
Relative Azimuth Angle
Albedo(surface)
Latitude(Low, Mid, and High)
Total Column Ozone(Total ozone of vertical column of atmosphere)(200, 250, 300, 350)

[[Output]]
The radiance for 300-340 nm wavelength range.

