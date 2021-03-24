import numpy as np
import cv2 as cv
from sys import argv

class ImageProcessor():

    def __init__( self, image_path ):
        self.ROOT_WINDOWS = "main"
        cv.namedWindow( self.ROOT_WINDOWS )
        self.original_image = cv.imread( image_path, 0 )
        self.processed_image = self.original_image

    def run( self ):

        while True:
            self.show()
            code = cv.waitKey(1)

            if( code == ord('q') ):
                cv.destroyAllWindows()
                break
            elif( code == ord('r') ):
                self.reset_image()
            elif( code == ord('n') ):
                self.to_negatif()
            elif( code == ord('h') ):
                self.hist_eq()
            elif( code == ord('l') ):
                self.log_transform()
            # elif( code == ord('p') ):
            #     self.power_law()
            elif( code == ord('m') ):
                #use brain image
                self.median_filtering()
            elif( code == ord('c') ):
                #use  xray_penuomia
                self.contrast_streching()

    def show( self ):
        cv.imshow( self.ROOT_WINDOWS, self.processed_image )

    def reset_image( self ):
        self.processed_image = self.original_image

    def to_negatif( self ):
        self.processed_image = 255.0 - self.processed_image

    def hist_eq( self ):
        self.processed_image = cv.equalizeHist( self.processed_image )

    def log_transform( self ):
        self.processed_image = np.array( 100.0 * np.log10( 1.0 + self.processed_image ), dtype=np.uint8 )

    # def power_law( self ):
    #     first_gamma = np.array( 255.0 * ( self.processed_image / 255.0 ) ** 0.3, dtype=np.uint8 )
    #     second_gamma = np.array( 255.0 * ( self.processed_image / 255.0 ) ** 0.6, dtype=np.uint8 )
    #     self.processed_image = cv.hconcat( [ first_gamma, second_gamma ] )

    def median_filtering( self ):
        median = cv.medianBlur(self.processed_image, 5)
        self.processed_image = np.concatenate((self.processed_image, median), axis=1) #membandingkan

    def contrast_streching( self ):
        norm_img = cv.normalize(self.processed_image, None, alpha=0, beta=1.2, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        norm_img = np.clip(norm_img, 0, 1)
        norm_img = (255*norm_img).astype(np.uint8)
        self.processed_image = np.concatenate((self.processed_image, norm_img), axis=1) #membandingkan


def main():
    img_path = argv[1]
    processor = ImageProcessor( img_path )
    processor.run()

if __name__ == "__main__":
    main()