# Begin Installation - Ubuntu linux only
#    sudo pip install pafy
#    sudo pip install youtube-dl
#    sudo apt-get install python-opencvsudo apt-get install python-opencv    # Note opencv 2.4.8 
# End Installation


# import the necessary packages
import cv2
import os
import pafy
import pathlib2

class ImagesYoutubeExtract(object):
    def __init__(self , images_save_location):
        self.images_save_location = images_save_location

    def get_urls_search_query(self, query):
        import urllib
        import urllib2
        from bs4 import BeautifulSoup

        textToSearch = query
        query = urllib.quote(textToSearch)
        url = "https://www.youtube.com/results?search_query=" + query
        response = urllib2.urlopen(url)
        html = response.read()
        soup = BeautifulSoup(html)
        query_urls = []
        orig_urls = []
        for vid in soup.findAll(attrs={'class':'yt-uix-tile-link'}):
            urlorig = 'https://www.youtube.com' + vid['href']
            url = 'https://www.youtube.com' + vid['href']
            url = url.replace("watch?v=", "embed/")
            if "embed/" in url :
                query_urls.append(url)
                orig_urls.append(urlorig)
        return query_urls[1:] , orig_urls[1:]

    def extract_images_youtube(self, youtube_url , query):
        downloadable_video_url = self._get_video_url_replicate_youtube(youtube_url)
        print(downloadable_video_url)
        self._get_image_frame_from_video(downloadable_video_url , query)

    def _get_video_url_replicate_youtube(self, youtube_url):
        video = pafy.new(youtube_url)
        return video.getbest().url
    

    def _get_image_frame_from_video(self, video_url , query):
        
        # Playing video from file:
        vidcap = cv2.VideoCapture(video_url)
        total_frame = int(vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        desired_num_frames = int(total_frame/20)
        fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
        videotime = total_frame / fps

        # Capture frame-by-frame
        success, frame = vidcap.read()
        success = True
        currentFrame = 0

        save_path = os.path.join(self.images_save_location, query)
        if not os.path.exists(save_path):
            pathlib2.Path(os.path.join(self.images_save_location, query)).mkdir(parents=True, exist_ok=True)
            while currentFrame*10000 <= videotime*1000:
                vidcap.set(cv2.cv.CV_CAP_PROP_POS_MSEC,(currentFrame*10000)) #extract frame every ten seconds
                # Saves image of the current frame in jpg file
                filename = os.path.join(save_path , "frame" + str(currentFrame) + '.jpg' )
                print ('Creating...' + filename)
                cv2.imwrite(filename, frame)
                success, frame = vidcap.read()

                # To stop duplicate images
                currentFrame += 1
        else: 
            print ('Current query videos already extracted')        

        # When everything done, release the capture
        vidcap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    images_save_location = "/var/www/img-search-cnn/webapp/dataset/applicationData"
    youtube_url = "https://www.youtube.com/watch?v=kQcUamGg7Yw"
    obj_iye = ImagesYoutubeExtract(images_save_location)

    urls , origurls = obj_iye.get_urls_search_query("dog") 
    print (origurls[0])
    obj_iye.extract_images_youtube(origurls[0] , "salgaris")
    #obj_iye.extract_images_youtube(youtube_url , "salgari")