CPPFLAGS = -g -Wall -DCHECK_OVERFLOW -O2
LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_objdetect -lopencv_imgcodecs -lopencv_video
PATH_INCLUDES = /opt/installation/OpenCV-3.4.4/include
PATH_LIB = /opt/installation/OpenCV-3.4.4/lib

OBJS_TB = main.o utils.o ShowManyImages.o ColorTracker.o
BIN_TB = main

all: link_all
	rm -f $(OBJS_TB)

link_all: $(OBJS_TB)
	g++ -o $(BIN_TB) $(OBJS_TB) -L$(PATH_LIB) $(LIBS)

main.o: main.cpp
	g++ $(CPPFLAGS) -I$(PATH_INCLUDES) -c main.cpp

utils.o: utils.cpp
	g++ $(CPPFLAGS) -I$(PATH_INCLUDES) -c utils.cpp

ShowManyImages.o: ShowManyImages.cpp
	g++ $(CPPFLAGS) -I$(PATH_INCLUDES) -c ShowManyImages.cpp

ColorTracker.o: ColorTracker.cpp
	g++ $(CPPFLAGS) -I$(PATH_INCLUDES) -c ColorTracker.cpp

clean:
	rm -f $(BIN_TB) $(OBJS_TB)