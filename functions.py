from libraries import *

# Function to read and parse the kml file and store the polygon coordinates
def read_kml(kml_file):
    """Reads a kml file and returns the coordinates of the polygons in it"
    Args: kml_file (str): The path to the kml file
    """
    tree = etree.parse(kml_file)
    root = tree.getroot()
    namespace = '{http://www.opengis.net/kml/2.2}'
    placemarks = root.findall(f'.//{namespace}Placemark')
    polygon_coords = list()
    for placemark in placemarks:
        polygon = placemark.find(f'.//{namespace}Polygon')
        if polygon is not None:
            coord_text = polygon.find(f'.//{namespace}coordinates').text
            coords = [tuple(map(float, coord.split(',')))[:2] for coord in coord_text.strip().split(' ')]
            polygon_coords.append(coords)
    return polygon_coords

# Function to check if a point is in the polygon
def is_point_in_mask(polygon_coords, point):
    """Checks if a point is in the polygon
    Args:
        polygon_coords (list): The coordinates of the polygon
        point (tuple): The coordinates of the point
    Output:
        bool: True if the point is in the polygon, False otherwise
    """

    polygon = Polygon(polygon_coords[0])
    return polygon.contains(Point(point))

# Function to parse NC files
def data_parsing(file):
    """Reads a netCDF file for the track and returns the coordinates of the points in it.
    Args: 
        file (str): The path to the netCDF file
    Output:
        points (list): The coordinates for each measurement in the track
        lon (list): The longitude of the points in the track
        lat (list): The latitude of the points in the track
        pwr (list): The power waveform of the signal.
        time (list): The time the time at which the middle of the averaged group of pulses impacts the ground
        altitude (list): The altitude of the satellite at the time of the measurement
        window_delay (list): Calibrated 2-way window delay: distance from CoM to middle range window
    """
    points=list()
    with netCDF4.Dataset(file, 'r') as nc:
        lon = nc.variables['lon_20_ku'][:]
        lat = nc.variables['lat_20_ku'][:]
        pwr=nc.variables['pwr_waveform_20_ku'][:]
        time=nc.variables['time_20_ku'][:]
        altitude=nc.variables['alt_20_ku'][:]
        altitude_rate=nc.variables['orb_alt_rate_20_ku'][:]
        window_delay=nc.variables['window_del_20_ku'][:]
        # Get the range corrections
        ocean_load_tide = nc.variables['ocean_tide_01'][:]
        solid_earth_tide = nc.variables['solid_earth_tide_01'][:]
        geocentric_polar_tide = nc.variables['pole_tide_01'][:]
        dry_tropospheric_correction = nc.variables['mod_dry_tropo_cor_01'][:]
        wet_tropospheric_correction = nc.variables['mod_wet_tropo_cor_01'][:]
        ionospheric_correction = nc.variables['iono_cor_gim_01'][:]
        ionospheric_correction_2 = nc.variables['alt_avg_01_ku'][:]

        for latitude, longitude in zip(lat,lon):
            points.append([longitude, latitude])
    return points, lon, lat, pwr, time, altitude, window_delay

# Calculate Earth Radius at certain points
def earth_radius(point):
    # Since Earth can be considered an oblate spheroid
    a=6378.137*10**3    # Earth radius at the equator in meters
    b=6356.7523*10**3   # Earth radius at the poles in meters
    #flat_coeff=(a-b)/a
    #e=math.sqrt(2*flat_coeff-flat_coeff**2)     # Eccentricity
    #x=math.radians(point[1])
    #y=math.radians(point[0])
    # Started doing it manually and then I found this formula and adjusted it
    R = math.sqrt((((a**2)*math.cos(point[0]))**2+((b**2)*math.sin(point[0]))**2)/((a*math.cos(point[0]))**2+(b*math.sin(point[0]))**2))
    return R

# Implementing the TFMR algorithm
def TFMR(waveform,operating_mode):
    """Implements the TFMR algorithm to detect the POCA response in the power waveform
    Args:
        waveform (np.array): The power waveform of the signal
        operating_mode (str): The operating mode of the satellite (LRM or SAR)
    Output:
        nret: The index in the power waveform array which corresponds to POCA
    """
    # Step 1: Normalize waveform to its maximum
    noise_threshold=0.15
    waveform = waveform/ np.max(waveform)
    # Step 2: Calculate the thermal noise using the first n range bins
    n = 6
    thermal_noise = np.mean(waveform[:n])
    # Step 3: Flag bad data with P_N > T_N
    if thermal_noise > noise_threshold:
        return None
    # Step 4: Over-sample waveform by a factor of 10 using linear interpolation
    oversample_factor = 10
    interp_fn = interp1d(np.arange(waveform.size), waveform, kind='linear')
    interp_waveform = interp_fn(np.linspace(0, waveform.size - 1, waveform.size * oversample_factor))
    # Step 5: Calculate the smoothed waveform with a boxcar average of width 15
    smoothed_waveform = np.convolve(interp_waveform, np.ones(15) / 15, mode='valid')
    # Step 6: Calculate first derivative dP using a 3-point Lagrangian interpolation
    dP = np.zeros(smoothed_waveform.size)
    for i in range(1, smoothed_waveform.size - 1):
        dP[i] = (smoothed_waveform[i + 1] - smoothed_waveform[i - 1]) / 2
    # Step 7: Determine the first local maxima using dP and the criteria P > P_N + T_N
    Pmax1 = None
    for i in range(1, dP.size - 1):
        if smoothed_waveform[i] > thermal_noise + noise_threshold and dP[i] > dP[i - 1] and dP[i] > dP[i + 1]:
            Pmax1 = smoothed_waveform[i]
            break
    if Pmax1 is None:
        return None
    # Step 8: Determine the location n̂ of the first gate exceeding the threshold level TL
    if operating_mode=='LRM':
        TL = 0.25
    else:
        TL = 0.4    
    for i in range(1, interp_waveform.size):
        if interp_waveform[i] > Pmax1 * TL + thermal_noise:
            nhat = i
            break
    # Step 9: Determine the leading edge position nret by interpolation between adjacent oversampled bins to the threshold crossing
    nret = (nhat - 1) + (TL - interp_waveform[nhat - 1]) / (interp_waveform[nhat] - interp_waveform[nhat - 1])
    nret = int(np.round(nret / oversample_factor))  # Convert to original sampling rate and determine the integer part
    return nret

# Range equation
def range_eq(Tw,n,radar_mode):
    """Calculates the range of the POCA point using the range equation
    Args:
        Tw (float): The window delay converted to seconds
        n (int): The index of the POCA point in the power waveform
        radar_mode (str): The operating mode of the satellite (LRM or SAR)
    Output:
        R_sat (float): The range of the POCA point in meters
    """

    B=32*10**7  # Hz   Chirp Bandwidth
    c=299792458 # m/s   Speed of light
    if radar_mode=='SAR' or radar_mode=='LRM':
        N_s=128    # Number of samples
        if radar_mode=='SAR':
            R_sat=(c*Tw/2)-(N_s*c/(4*B))+n*c/(2*B)
            
        else:
            R_sat=(c*Tw/2)-(N_s*c/(4*B))+n*c/(2*B)
    else:   # LRM mode
        N_s=256
        R_sat=(c*Tw/2)-(N_s*c/(4*B))+n*c/(2*B)
    return R_sat

# Swath width equation
def swath_width_LRM(alt):
    beam_along=1.06 # degrees   Beam angle along track
    beam_across=1.1992 # degrees   Beam angle across track
    along_track=2*alt*math.tan(beam_along/2*math.pi/180) # m   Along track swath width
    print(f'along_track is: {along_track:.2f}')
    across_track=2*alt*math.tan(beam_across/2*math.pi/180) # m   Across track swath width
    print(f'across_track is: {across_track:.2f}')
    swath_width_area=0.5*along_track*across_track*math.pi # m^2   Swath area
    return swath_width_area

def filter_elevation_data(data, max_threshold=3700, spike_threshold=20):
    """Filters the elevation data to remove outliers
    Args:
        data (ndarray): The elevation data of the DEM
        max_threshold (int): The maximum elevation threshold in meters
        spike_threshold (int): The maximum elevation difference between two adjacent pixels in meters
    Output:
        filtered_data (ndarray): The filtered elevation data
    """
    filtered_data = np.zeros_like(data)
    # Check if the elevation values are within the max threshold range
    within_threshold = data <= max_threshold
    # Check if the elevation values are outliers due to sudden spikes
    within_spike_threshold = np.abs(data[:, 1:] - data[:, :-1]) <= spike_threshold
    # Create the mask based on the conditions
    mask = np.concatenate((within_threshold[:, :1], within_threshold[:, 1:] & within_spike_threshold), axis=1)
    # Apply the mask to filter the elevation data
    filtered_data[mask] = data[mask]
    return filtered_data

#############################################################################################################################
######### DEM PROCESSING ####################################################################################################

def crop_dem_Greenland(input_dem):        # We crop the DEM to get Greenland, or else it is huge for interpolation
    """Crops the DEM to get only Greenland
    Args:
        input_dem (str): The path to the DEM file
    Output:
        output_file (str): The path to the cropped DEM file
    """

    data=gdal.Open(input_dem)
    gt=data.GetGeoTransform()
    output_file = 'Greenland_dem.tif.tif'
    out_ds = gdal.Translate(
    output_file,
    data,
    srcWin=[6500,8500,10000-6500,15000-8500],       # Cropy by (y,x)
)
    out_ds = None       # We always need to explicitly close the dataset for the changes to be saved.
    ds = None
    return output_file

def highest_point_in_swath(file, point, swath_area, aspect_ratio=1.0 ): 
    # Open the input DEM file
    #input_dem = gdal.Open(file)

    # Create an in-memory GDAL dataset for the resampled DEM
    #output_dem = "/vsimem/resampled_dem.tif"

    # Perform the resampling using the specified method and output resolution
    #gdal.Warp(output_dem, input_dem, xRes=x_res, yRes=y_res, resampleAlg=resampling_method)

    # Open the resampled DEM dataset
    #dataset=gdal.Open(output_dem)
    dataset=gdal.Open(file)
    # Close the input DEM file
    #input_dem = None
    # We store in different variables the different chromatic channels
    band1=dataset.GetRasterBand(1) # Monochrome channel
    b1=band1.ReadAsArray()
    # Cropping to get only Greenland
    b1=b1[8500:15000,6500:10000]    
    b1=np.where(b1<0,0,b1)
    GT=dataset.GetGeoTransform()
    y,x=dataset.RasterYSize,dataset.RasterXSize
    # 2. Import the coordinates of a point in a tuple (longitude, latitude)
    point_lon, point_lat = point
    # 3. Import the area of the ellipse and calculate the axes a and b
    a = np.sqrt(swath_area / (np.pi * aspect_ratio))
    b = a * aspect_ratio
    # 4. Create X and Y meshgrids from the DEM file using the GDAL library
    x_ref = np.arange(dataset.RasterXSize) * GT[1] + GT[0]
    y_ref = np.arange(dataset.RasterYSize) * GT[5] + GT[3]
    x_dem, y_dem = np.meshgrid(x_ref, y_ref)
    # 5. Convert the longitude and latitude coordinates of the imported "Point" into the X and Y coordinate system of the DEM file
    point_x, point_y = point_lon, point_lat
    point_col = int((point_x - GT[0]) / GT[1])
    point_row = int((point_y - GT[3]) / GT[5])

    # 6. Create a meshgrid for the swath area, with the center being the imported point ("Point")
    x_rel = np.linspace(-a, a, 100)
    y_rel = np.linspace(-b, b, 100)
    x_swath_rel, y_swath_rel = np.meshgrid(x_rel, y_rel)
    x_swath = x_swath_rel + point_x
    y_swath = y_swath_rel + point_y

    # 7. Overlay the meshgrid with the X and Y grids, matching their coordinates
    swath_mask = (x_swath_rel / a) ** 2 + (y_swath_rel / b) ** 2 <= 1
    rows, cols = np.argwhere((y_dem == y_swath[:, None]) & (x_dem == x_swath)).T        # Here I get an error if I dont interpolate the DEM
    elev_swath = np.full_like(x_swath, np.nan)
    elev_swath[swath_mask] = b1[rows, cols]
    #elev_swath[swath_mask] = b1[y_dem == y_swath[:, None], x_dem == x_swath]

    # 8. Calculate the distance of the highest point in the swath area from the center of the grid
    highest_index = np.unravel_index(np.nanargmax(elev_swath), elev_swath.shape)
    highest_x, highest_y = x_swath[highest_index], y_swath[highest_index]
    highest_point = np.array([highest_x, highest_y])
    center_point = np.array([point_x, point_y])
    dist = np.linalg.norm(highest_point - center_point)
    # 9. Convert the X and Y coordinates of the highest point in longitude and latitude
    highest_lon, highest_lat = highest_x, highest_y
    return highest_lon, highest_lat, dist

#############################################################################################################################
#############################################################################################################################

# Convert the DEM file into a dictionary
def DEM_dict(file_path):
    """
    This function reads a DEM file and returns a dictionary with the following keys:
    - elevations: a 2D array with the elevation values
    - lon_values: a 2D array with the longitude values
    - lat_values: a 2D array with the latitude values
    - crs: the coordinate system of the DEM file
    
    file_path: path to the DEM file
    """
    
    with rasterio.open(file_path) as dem:
        # Read the DEM file
        elevations = dem.read(1)
        elevations = np.where(elevations < 0, 0, elevations)

        # Get the longitude and latitude values
        lon_values, lat_values = np.meshgrid(
            np.linspace(dem.bounds.left, dem.bounds.right, dem.width),
            np.linspace(dem.bounds.bottom, dem.bounds.top, dem.height)
        )

        # Get the coordinate system
        crs_dem = dem.crs
        print(f'DEM coordinate system: {crs_dem}')

        # Transform the coordinates to degrees
        crs_degrees = pyproj.CRS.from_string('EPSG:4326')
        transformer = pyproj.Transformer.from_crs(crs_dem, crs_degrees, always_xy=True)
        lon_deg, lat_deg = transformer.transform(lon_values, lat_values)

        # Create the dictionary
        dem_dict = {
            "lon": lon_deg,
            "lat": lat_deg,
            "z": elevations
        }

        return dem_dict


# Visualize the track of the file, along with the elevations in the map of Greenland

def visualize_elevations_with_track(DEM, lon, lat, save_filename, elevation_array):
    """
    Visualize elevations from a DEM along with the track of a satellite.

    Args:
        DEM (dict): A dictionary containing latitude (DEM['lat']), longitude (DEM['lon']).
        lon (array-like): Array of longitude values for the satellite track.
        lat (array-like): Array of latitude values for the satellite track.
        save_filename (str, optional): Filename to save the plot. If not provided, the plot is displayed instead.
        elevation_array (array-like): 1-D array of elevation values.

    Output:
        A plot of the elevations from the DEM along with the satellite track.
    """
    # Plot the DEM
    plt.imshow(DEM['z'], extent=(DEM['lon'].min(), DEM['lon'].max(), DEM['lat'].min(), DEM['lat'].max()), cmap='terrain', origin='upper', aspect='auto')
    
    # Plot the satellite track with elevations
    plt.scatter(lon, lat, c=elevation_array, cmap='rainbow', vmin=elevation_array.min(), vmax=elevation_array.max())

    # Add a colorbar
    cbar = plt.colorbar()
    cbar.set_label('Elevation')

    # Set x-axis and y-axis labels
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Set plot title
    plt.title('Elevations with Satellite Track')

    # Save the plot or display it
    if save_filename:
        plt.savefig(save_filename)
    else:
        plt.show()


# Nadir to POCA
def NadirToPOCA(DEM_dict,lon,lat,dxx,dyy,altitude,footprint,mt,sm):
    """
    This function converts the nadir coordinates of a satellite to the POCA coordinates.
    It also returns the x and y offset coordinates between the nadir and the POCA coordinates.       
    """

    # Find the appropriate hemisphere where the track is
    if np.mean(lat) >= 0:
        lathemisphere = 90.
    else:
        lathemisphere = -90.
    # Create the stereographic projection
    stereographic_projection = pyproj.Proj(proj='stere', ellps='WGS84', lat_0=lathemisphere, lon_0=np.mean(lon), lat_ts=np.mean(lat), x_0=np.min(lon), y_0=np.min(lat), k=1)
    # Convert the coordinates of DEM points to the stereographic projection, k=1)
    print(f'Step: Stereographic projection: Complete.')
    # Convert the coordinates of DEM points to the stereographic projection
    DEM_x, DEM_y= stereographic_projection(DEM_dict['lon'], DEM_dict['lat'])
    print(f'Step: DEM_x, DEM_y to stereographic projection: Complete.')
    # Create 2D grids based on the min and max points of DEM_x and DEM_y with a step size of "dxx" and "dyy", respectively.
    xx, yy = np.meshgrid(np.arange(np.min(DEM_x), np.max(DEM_x) + dxx, dxx),np.arange(np.min(DEM_y), np.max(DEM_y) + dyy, dyy))
    print(f'Step: xx, yy meshgrids: Complete.')
    #print(f'Shape of DEM_x: {np.shape(DEM_x)}\nShape of DEM_y: {np.shape(DEM_y)}\nShape of xx: {np.shape(xx)}\nShape of yy: {np.shape(yy)}\nShape of DEM[z]: {np.shape(DEM_dict["z"])}')

    # Formatting the input variables for the griddata function 
    elev = DEM_dict['z'].ravel()
    points = (DEM_x.ravel(), DEM_y.ravel())  # This is now a tuple of 1D arrays
    grid_points = (xx.ravel(), yy.ravel())  # This is now a tuple of 1D arrays
    print(f'Step: points, values, grid_points: Complete.')
    # Interpolate the DEM data onto the offset grid
    surf = spint.griddata(points, elev, grid_points, method=mt)
    # Reshaping to get 2-D grid because my gaussian filter function only takes 2-D array
    surf = surf.reshape(xx.shape)
    print(f'Step: Surf interpolation: Complete.')

    # Convert nadir latitude and longitude to stereographic projection
    Px, Py = stereographic_projection(lon, lat)
    print(f'Step: Px, Py to stereographic projection: Complete.')
    # Interpolate the DEM elevation data onto the offset grid
    #h = spint.griddata((DEM_x, DEM_y), DEM_dict['z'], (Px, Py), method=mt)
    h = spint.griddata(points, elev, (Px, Py), method=mt)
    print(f'Step: h interpolation: Complete.')
    # Compute size parameters for Gaussian filter based on smoothing distance and grid steps
    sizex = int(sm / dxx)
    print(f'Size_x: {sizex}')
    sizey = int(sm / dyy)
    print(f'Size_y: {sizey}')
    print(f'Length of surf: {len(surf)}')
    print(f'Step: sizex, sizey: Complete.')
    # Smooth the DEM data with a Gaussian filter
    demSmooth = gaussian_filter(surf, sigma=[sizex, sizey])
    xxSmooth = gaussian_filter(xx, sigma=[sizex, sizey])
    yySmooth = gaussian_filter(yy, sigma=[sizex, sizey])
    print(f'Step: demSmooth, xxSmooth, yySmooth: Complete.')
    # Determine the size of the search window based on the footprint radius and step size
    stepx = int(footprint / dxx)
    stepy = int(footprint / dyy)
    # Find maximum indices for xxSmooth array
    maxx = xxSmooth.shape[1]
    maxy = xxSmooth.shape[0]
    # Dummy distance for offsetting indices: used to extend the search window around each nadir point when looking for potential POCAs ()
    dummy_dist = 50    
    # Initialize the new x and y coordinate arrays
    Newx = lat * np.NaN
    Newy = lat * np.NaN

    print(f'Looping over each nadir point...')
    count=0
    # Loop over each nadir point
    for ii in range(len(lat)):
        if ~np.isnan(h[ii]):
            # The next several lines are defining a region around the current point within which the function will search for other points.
            # Compute a start and stop index for both x (longitude) and y (latitude) based on the footprint radius, the current point's position, and the 'dummy distance'.
            idx = int(abs(Px[ii] - xxSmooth[0, 0]) / dxx)
            xstart = np.max([idx - (stepx + dummy_dist), 0])
            xstop = np.min([idx + (stepx + dummy_dist), maxx])
            idy = int(abs(Py[ii] - yySmooth[0, 0]) / dyy)
            ystart = np.max([idy - (stepy + dummy_dist), 0])
            ystop = np.min([idy + (stepy + dummy_dist), maxy])

            # Selecting the region of the smoothed DEM within the window defined by the start and stop indices for x and y.
            Newsurf = demSmooth[ystart:ystop, xstart:xstop]  # .compress(kernel.ravel())
            # Selecting the corresponding regions of the smoothed coordinate arrays.
            Newxx = xxSmooth[ystart:ystop, xstart:xstop]  # .compress(kernel.ravel())
            Newyy = yySmooth[ystart:ystop, xstart:xstop]  # .compress(kernel.ravel())
            # Create a mask of the same shape as the selected region, where True values indicate points within the footprint radius of the current point and also where the DEM is not NaN
            kernel = (((Px[ii] - Newxx) ** 2 + (Py[ii] - Newyy) ** 2) < footprint ** 2) & ~np.isnan(Newsurf)
            # Apply the kernel to the selected regions, filtering out points outside the footprint radius and where the DEM is NaN
            Newsurf = Newsurf[kernel]
            Newxx = Newxx[kernel]  # .compress(kernel.ravel())
            Newyy = Newyy[kernel]  # .compress(kernel.ravel())
            # Calculate the squared Euclidean distance from the current point to each point in the filtered region
            dist = (Newxx - Px[ii]) ** 2 + (Newyy - Py[ii]) ** 2
            # Calculate a sort of "adjusted" distance, taking into account both the distance on the ground and the difference in altitude between the current point and each point in the filtered region.
            dist_sat = (altitude - (Newsurf - h[ii])) ** 2 + dist
            # Create and apply a boolean mask where True values correspond to points in the filtered region that have the minimum adjusted distance.
            id = (dist_sat.min() == dist_sat)
            dumx = Newxx[id]
            dumy = Newyy[id]
            # Check if there is more than one point with the minimum adjusted distance, and select the one with the minimum ground distance.
            if len(dumx) > 1:
                tmp = dist[id]
                id2 = (tmp.min() == tmp)
                dumx = dumx[id2]
                dumy = dumy[id2]
            Newx[ii] = dumx
            Newy[ii] = dumy
        else:
            Newx[ii] = np.NaN
            Newy[ii] = np.NaN
        count += 1
        print(f'Completed: {count/len(lat)*100:.2f}%')
    # Convert the output coordinates back from the stereographic projection to latitude and longitude.
    Newlon, Newlat = stereographic_projection(Newx, Newy, inverse=True)
    
    return Newx, Newy, Newlon, Newlat

# Create the grid file
def makeRelocationTemp(RLoutdir, Utilfile, res, sat, fp):

    """

    :param RLoutdir:    Directory for the output file

    :param Utilfile:    The Util file with DEM

    :param res:         resolution of the estimated syntetic data

    :param sat:         altitude of the satellite in km

    :param fp:          footprint size of the satellite

    :return:            Make a file named Relocation_"res"km_fp"fp"'km_alt"sat"km.nc

    """

    from netCDF4 import Dataset

    import os

    import pyproj

    import numpy as np


    # Define standad projection for the polar stenographic projection

    Pollat0 = 90.

    Pollon0 = -45.

    Pollatts = 70.

    ell = 'WGS84'

    PolProj = pyproj.Proj(proj='stere', ellps=ell, lat_0=Pollat0, lon_0=Pollon0, lat_ts=Pollatts, x_0=0, y_0=0, k_0=1.0)


    lllat = 58  # lower left lat

    lllon = -57  # Lower left lon

    urlat = 81  # upper right lat

    urlon = 20  # upper right lon

    xll, yll = PolProj(lllon, lllat)

    xur, yur = PolProj(urlon, urlat)

    x = np.arange(xll, xur, res * 1000.)

    y = np.arange(yll, yur, res * 1000.)

    nx, ny = np.meshgrid(x, y)

    nlon, nlat = PolProj(nx, ny, inverse=True)


    data = Dataset(Utilfile)

    UtilLat = data.variables["Lat"][:]

    UtilLon = data.variables["Lon"][:]

    UtilDEM = data.variables["DEM_Arctic"][:]

    data.close()


    DEM = {'lat': UtilLat.ravel(), 'lon': UtilLon.ravel(), 'z': UtilDEM.ravel()}


    sm = 2000.

    Newx, Newy, Newlat, Newlon = NadirToPOCA(DEM, nlat.ravel(), nlon.ravel(), sat * 1000., fp * 1000., 100., 100., sm,

                                          'cubic')


    RLoutName = RLoutdir + os.sep + 'Relocation_' + str(int(res)) + 'km_fp' + str(int(fp)) + 'km_alt' + str(

        int(sat)) + 'km.nc'

    nc = Dataset(RLoutName, 'w', format='NETCDF4')

    nc.setncatts({'Title': 'Syntetic relocation file', \

                  'summary': 'A satellite at %s km, footprint of %s km and DEM smoothing of %s km' % (

                  str(int(sat)), str(int(fp)), str(int(sm / 1000.)),) \

                  })


    ync = nc.createDimension('l', len(Newx))


    var1 = nc.createVariable('nx', 'f4', ('l'), zlib=True)

    var1.long_name = 'Nadir x'

    var1[:] = nx.ravel()


    var1 = nc.createVariable('ny', 'f4', ('l'), zlib=True)

    var1.long_name = 'Nadir y'

    var1[:] = ny.ravel()


    var1 = nc.createVariable('nlat', 'f4', ('l'), zlib=True)

    var1.long_name = 'Nadir lat'

    var1[:] = nlat.ravel()


    var1 = nc.createVariable('nlon', 'f4', ('l'), zlib=True)

    var1.long_name = 'Nadir lon'

    var1[:] = nlon.ravel()


    var1 = nc.createVariable('px', 'f4', ('l'), zlib=True)

    var1.long_name = 'poca x'

    var1[:] = Newx


    var1 = nc.createVariable('py', 'f4', ('l'), zlib=True)

    var1.long_name = 'poca y'

    var1[:] = Newy


    var1 = nc.createVariable('plat', 'f4', ('l'), zlib=True)

    var1.long_name = 'poca lat'

    var1[:] = Newlat


    var1 = nc.createVariable('plon', 'f4', ('l'), zlib=True)

    var1.long_name = 'poca lon'

    var1[:] = Newlon


    nc.close()

    return