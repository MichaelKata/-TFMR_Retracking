# Import the libraries
import functions as fn
import matplotlib.pyplot as plt
from libraries import *

start=time.time()
# Import the file
file_list=list()
with open('Valid_Names.txt', 'r') as file:
    for line in file:
        file_list.append(line.rstrip('\n'))
if 'LRM' in file_list[0]:
    mode='LRM'
else:
    mode='SAR'

# Get the necessary variables from the needed file
track_file=file_list[0]
points,lon,lat,pwr,time_meas,altitude,window_delay=fn.data_parsing(track_file)  # The code can be modified to loop through all the files in the list

# Import elevation map and get the elevations, the latitude and the longitude
dem_file='arcticdem_mosaic_500m_v3.0.tif'
print(f'Cropping the DEM')
cropped_dem=fn.crop_dem_Greenland(dem_file)

# Use the function to get the DEM into a dictionary form dictionary
DEM = fn.DEM_dict(cropped_dem)   

filtered_elevations = fn.filter_elevation_data(DEM['z'])
DEM['z'] = filtered_elevations

########## Function to calculate the range from satellite to POCA ##########
def process_power(power):
    """Function to calculate the range from satellite to POCA
    Args:
        power (float): The power waveform of the measurement
    Output:
        range_measurement (float): The range from satellite to POCA
    """
    power_index = np.where(pwr == power)[0][0]
    idx = fn.TFMR(power,mode)
    if idx==None:
        print(f'Power is lower than the threshold for measurement number {points.index(points[power_index])+1}')
        return np.nan # If the TFMR is not found, then the power is too low
    range_measurement = fn.range_eq(window_delay[power_index], idx, mode)
    return range_measurement

# ΜULTIPROCESSING FOR THE WHOLE FILE TO GET THE POCA RANGES
def worker(i):
    range_measurement = process_power(pwr[i])
    return range_measurement, i

def process_result(result):
    range_measurement, i = result
    if range_measurement is not None and not np.isnan(range_measurement):
        #print(f'The range from satellite to POCA for measurement number {points.index(points[i])+1} is: {range_measurement:.2f}m')
        return range_measurement, points[i], time_meas[i], altitude[i]
    return None

if __name__ == '__main__':
    num_processes = 8  # Adjust the number of processes based on your machine's capabilities
    range_measurements = []
    coords_final = []
    time_final = []
    altitude_final = []
    print(f'Starting multiprocessing with {num_processes} threads.')

    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(worker, range(len(pwr)))

        for result in results:
            processed = process_result(result)
            if processed is not None:
                range_measurement, coord, time_final_i, altitude_i = processed
                range_measurements.append(range_measurement)
                coords_final.append(coord)
                time_final.append(time_final_i)
                altitude_final.append(altitude_i)
                
    coords_final = np.array(coords_final)  
    range_measurements = np.array(range_measurements)
# Find the POCA coordinates
satellite_altitude = altitude[0]
sat_lon = coords_final[:,0]
sat_lat = coords_final[:,1]
footprint_radius = 6000
d_xx=250
d_yy=250
smoothing_distance=500
interpol='nearest'
print(f'Calculating the POCA coordinates')
new_x, new_y, new_lon, new_lat = fn.NadirToPOCA(DEM,sat_lon,sat_lat,d_xx,d_yy,satellite_altitude,footprint_radius,interpol,smoothing_distance)

np.savetxt('range coordinates.txt', points, fmt='%2f', delimiter='\t')

print(f'Number of range measurements: {len(range_measurements)}')
print(f'Number of POCA coordinates: {len(new_lon)}')
# Save the coordinates of the nadir and the POCA for this track

print(f'Saving the nadir coordinates')
np.savetxt('nadir_coordinates.txt', coords_final, fmt='%2f', delimiter='\t')
print(f'Saving the POCA coordinates')
np.savetxt('POCA_coordinates.txt', np.c_[new_lon,new_lat], fmt='%2f', delimiter='\t')
#fn.visualize_elevations_with_track(DEM,new_lon,new_lat,'POCA_coordinates')

d_nadir_poca = np.sqrt(np.sum((coords_final - np.column_stack((new_lon, new_lat)))**2, axis=1))

# Calculate the range from satellite to nadir
elevations=altitude_final-np.sqrt(range_measurements**2-d_nadir_poca**2)

np.savetxt('Output_Data.txt', np.column_stack((elevations, coords_final, np.c_[new_lon,new_lat], time_final)),
           delimiter='\t', header='Elevations\tNadir Coordinates\tPOCA coordinates\tTime of measurement')


# Visualize the elevations with the track
fn.visualize_elevations_with_track(DEM,new_lon,new_lat,'Elevations at POCA', elevations)

end=time.time()
print(f'Elapsed time: {end-start:.2f}s')
