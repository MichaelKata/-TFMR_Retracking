from functions import read_kml, is_point_in_mask
import netCDF4
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

start_time = time.time()
def process_file(file_path):
    valid_nc_files = []
    with netCDF4.Dataset(file_path, 'r') as nc:
        lon = nc.variables['lon_20_ku'][:]
        lat = nc.variables['lat_20_ku'][:]
        pwr = nc.variables['pwr_waveform_20_ku'][:]

    for longitude, latitude in zip(lon,lat):
        points = [(longitude, latitude)]

    # Check if there are points from the file inside the mask (If the track goes through the mask)
    for point in points:
        if is_point_in_mask(polygon_coords, point):
            valid_nc_files.append(file_path)
    return valid_nc_files

if __name__ == '__main__':
    # Open, parse the files and create the polygon mask
    kml_file = 'Greenland_LRM_Mask.kml'
    polygon_coords = read_kml(kml_file)

    # Check if the track overlaps with the mask.
    folder_path = input('Enter the folder path: ')

    # Find all the valid .nc files in the folder
    nc_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.nc')]

    # Use all available CPU cores to process the files in parallel
    with Pool(4) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(process_file, nc_files), total=len(nc_files), leave=False):
            results.append(result)

    # Flatten the list of results into a single list of valid file names
    valid_files = [file for sublist in results for file in sublist]

    # Write the valid file names to a text file
    with open('Valid_Names.txt', 'w') as f:
        f.write('\n'.join(valid_files))

    print(f'The number of the tracks that go through the mask is: {len(valid_files)}')
    #if valid_files:
    #    print(valid_files)

end_time = time.time()
total_time = end_time - start_time
print(f'Total processing time: {total_time:.2f} seconds')

