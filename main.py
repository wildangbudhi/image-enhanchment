import numpy as np
import cv2 as cv
from sys import argv

class ImageProcessor():

    def __init__( self, image_path ):
        self.ROOT_WINDOWS = "main"
        cv.namedWindow( self.ROOT_WINDOWS )
        self.original_image = cv.imread( image_path, 0 )
        self.processed_image = self.original_image

        print( self.processed_image )
        print( type( self.processed_image ) )
    
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

    
    def show( self ):
        cv.imshow( self.ROOT_WINDOWS, self.processed_image )
    
    def reset_image( self ):
        self.processed_image = self.original_image

    def to_negatif( self ):
        self.processed_image = 255 - self.processed_image
    
    def hist_eq( self ):
        self.processed_image = cv.equalizeHist( self.processed_image )
    
    def log_transform( self ):
        self.processed_image = 50 *  np.array( np.log10( 1 + self.processed_image ), dtype=np.uint8 )

def main():
    img_path = argv[1]
    processor = ImageProcessor( img_path )
    processor.run()

if __name__ == "__main__":
    main()