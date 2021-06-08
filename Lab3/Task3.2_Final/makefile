FLAGS = -g -Wall -DCHECK_OVERFLOW -O2
LIBRARIES = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_objdetect -lopencv_imgcodecs -lopencv_video
INCLUDES = /opt/installation/OpenCV-3.4.4/include
LIBRARY_PATH = /opt/installation/OpenCV-3.4.4/lib
OTHER_FILES = main.o blobs.o ShowManyImages.o
MAIN_CPP = main

all: link_all
	rm -f $(OTHER_FILES)

link_all: $(OTHER_FILES)
	g++ -o $(MAIN_CPP) $(OTHER_FILES) -L$(LIBRARY_PATH) $(LIBRARIES)

main.o: main.cpp
	g++ $(FLAGS) -I$(INCLUDES) -c main.cpp

blobs.o: blobs.cpp
	g++ $(FLAGS) -I$(INCLUDES) -c blobs.cpp

ShowManyImages.o: ShowManyImages.cpp
	g++ $(FLAGS) -I$(INCLUDES) -c ShowManyImages.cpp

clean:
	rm -f $(MAIN_CPP) $(OTHER_FILES)
