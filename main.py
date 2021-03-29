import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sys import argv

class ImageProcessor():

    def __init__( self, image_path ):
        self.ROOT_WINDOWS = "main"
        cv.namedWindow( self.ROOT_WINDOWS )
        self.original_image = cv.imread( image_path, 0 )
        self.processed_image = self.original_image
        self.fig, ( self.ax1, self.ax2 ) = plt.subplots( 1, 2, figsize=( 9, 3 ) )
        self.fig.suptitle( "Histogram" )

    def run( self ):

        self.histogram_( self.original_image, self.processed_image )

        while True:
            self.show()
            code = cv.waitKey(1)

            if( code == ord('q') ):
                cv.destroyAllWindows()
                break
            elif( code == ord('r') ):
                self.reset_image()
            elif( code == ord('n') ):
                # use cancer
                self.to_negatif()
            elif( code == ord('c') ):
                #use xray_penuomia
                self.contrast_streching()
            elif( code == ord('m') ):
                #use brain image
                self.median_filtering()
            elif( code == ord('l') ):
                #use logaritmik
                self.log_transform()
            elif( code == ord('h') ):
                #use xray_penuomia, iris, bay
                self.hist_eq()

    def histogram_( self, original_image, equalized ):

        self.ax1.clear()
        self.ax2.clear()

        self.ax1.hist( original_image.ravel(), 256, [0, 256] )
        self.ax1.set_title( 'Source' )

        self.ax2.hist( equalized.ravel(), 256, [0, 256] )
        self.ax2.set_title( 'Result' )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.pause(0.0001)
        self.fig.show()

    def show( self ):
        preview_result = cv.hconcat( [ self.original_image, self.processed_image ] )
        cv.imshow( self.ROOT_WINDOWS, self.processed_image )

    def reset_image( self ):
        self.processed_image = self.original_image
        self.histogram_( self.original_image, self.processed_image )

    def to_negatif( self ):
        self.processed_image = 255 - self.processed_image
        self.histogram_( self.original_image, self.processed_image )

    def contrast_streching( self ):
        self.processed_image = cv.normalize(self.processed_image, None, alpha=0, beta=1.2, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        self.processed_image = np.clip(self.processed_image, 0, 1)
        self.processed_image = (255*self.processed_image).astype(np.uint8)
        self.histogram_( self.original_image, self.processed_image )

    def median_filtering( self ):
        self.processed_image = cv.medianBlur(self.processed_image, 5)
        self.histogram_( self.original_image, self.processed_image )

    def log_transform( self ):
        self.processed_image = np.array( 100.0 * np.log10( 1.0 + self.processed_image ), dtype=np.uint8 )
        self.histogram_( self.original_image, self.processed_image )

    def hist_eq( self ):
        self.processed_image = cv.equalizeHist( self.processed_image )
        self.histogram_(self.original_image, self.processed_image)

def main():
    img_path = argv[1]
    processor = ImageProcessor( img_path )
    processor.run()

if __name__ == "__main__":
    main()