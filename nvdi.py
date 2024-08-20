import numpy as np
import rasterio
from rasterio.enums import Resampling
from numba import jit, cuda
import logging
from sklearn.ensemble import IsolationForest
import cupy as cp

logging.basicConfig(filename='ndvi_calculation.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class RadiometricCalibration:
    ESUN = {
        'Band_29': 1549.0,
        'Band_40': 1044.0,
    }

    @staticmethod
    def calibrate(band_data, band_name, sun_elevation, gain=1.0, bias=0.0):
        sun_elev_rad = np.deg2rad(sun_elevation)
        radiance = (band_data.astype(float) * gain + bias) * RadiometricCalibration.ESUN[band_name] / np.cos(sun_elev_rad)
        return radiance

class AtmosphericCorrection:
    @staticmethod
    def correct(radiance, band_name):
        dark_object_value = np.percentile(radiance, 1)
        corrected_radiance = radiance - dark_object_value
        corrected_radiance[corrected_radiance < 0] = 0
        return corrected_radiance

class NDVI:
    @staticmethod
    @jit(nopython=True)
    def calculate_cpu(red_band, nir_band):
        ndvi_result = np.empty(red_band.shape, dtype=np.float32)
        for i in range(red_band.shape[0]):
            for j in range(red_band.shape[1]):
                numerator = nir_band[i, j] - red_band[i, j]
                denominator = nir_band[i, j] + red_band[i, j]
                ndvi_result[i, j] = -9999 if denominator == 0 else numerator / denominator
        return ndvi_result

    @staticmethod
    @cuda.jit
    def calculate_gpu(red_band, nir_band, ndvi_result):
        i, j = cuda.grid(2)
        if i < red_band.shape[0] and j < red_band.shape[1]:
            numerator = nir_band[i, j] - red_band[i, j]
            denominator = nir_band[i, j] + red_band[i, j]
            ndvi_result[i, j] = -9999 if denominator == 0 else numerator / denominator

class OutlierDetection:
    @staticmethod
    def detect_and_correct(ndvi_result):
        isolation_forest = IsolationForest(contamination=0.01)
        ndvi_reshaped = ndvi_result.reshape(-1, 1)
        outliers = isolation_forest.fit_predict(ndvi_reshaped)
        ndvi_result[outliers == -1] = -9999
        return ndvi_result

class HyperionProcessor:
    def __init__(self, input_file, output_file='ndvi_result.tif', sun_elevation=45.0):
        self.input_file = input_file
        self.output_file = output_file
        self.sun_elevation = sun_elevation

    def load_band(self, dataset, band_name):
        band_index = {'Band_29': 1, 'Band_40': 2}[band_name]
        return dataset.read(band_index)

    def validate_dimensions(self, red_band, nir_band):
        if red_band.shape != nir_band.shape:
            raise ValueError("Red and NIR bands must have the same dimensions.")

    def calculate_ndvi(self):
        logging.info("Starting NDVI calculation")
        
        with rasterio.open(self.input_file) as dataset:
            red_band = self.load_band(dataset, 'Band_29')
            nir_band = self.load_band(dataset, 'Band_40')
            
            self.validate_dimensions(red_band, nir_band)

            red_radiance = RadiometricCalibration.calibrate(red_band, 'Band_29', self.sun_elevation)
            nir_radiance = RadiometricCalibration.calibrate(nir_band, 'Band_40', self.sun_elevation)
            logging.info("Radiometric calibration completed")

            red_corrected = AtmosphericCorrection.correct(red_radiance, 'Band_29')
            nir_corrected = AtmosphericCorrection.correct(nir_radiance, 'Band_40')
            logging.info("Atmospheric correction completed")

            if red_band.size > 1e7:
                red_band_gpu = cp.array(red_corrected)
                nir_band_gpu = cp.array(nir_corrected)
                ndvi_result_gpu = cp.empty_like(red_band_gpu)
                NDVI.calculate_gpu[(16, 16), (16, 16)](red_band_gpu, nir_band_gpu, ndvi_result_gpu)
                ndvi_result = cp.asnumpy(ndvi_result_gpu)
                logging.info("NDVI calculated using GPU")
            else:
                ndvi_result = NDVI.calculate_cpu(red_corrected, nir_corrected)
                logging.info("NDVI calculated using CPU with Numba optimization")

            ndvi_result = OutlierDetection.detect_and_correct(ndvi_result)
            logging.info("Outlier detection and correction completed")

            self.save_ndvi_result(ndvi_result, dataset)
            logging.info(f"NDVI calculation complete. Output saved to {self.output_file}")
        
        return self.output_file

    def save_ndvi_result(self, ndvi_result, input_dataset):
        with rasterio.open(self.output_file, 'w', 
                           driver='GTiff', 
                           height=ndvi_result.shape[0], 
                           width=ndvi_result.shape[1], 
                           count=1, 
                           dtype=rasterio.float32,
                           crs=input_dataset.crs, 
                           transform=input_dataset.transform,
                           nodata=-9999) as dst:
            dst.write(ndvi_result, 1)
            logging.info("NDVI result saved to GeoTIFF")

if __name__ == '__main__':
    input_file = 'hyperion_data.e01'
    output_file = 'ndvi_result_advanced.tif'
    processor = HyperionProcessor(input_file, output_file)
    processor.calculate_ndvi()
