cmake_minimum_required (VERSION 3.10.2)

##### PLUGIN NAME #####
set(pluginname gipDarknet)


##### PLUGIN PATHS #####
set(PLUGIN_DIR ${PLUGINS_DIR}/${pluginname})
set(PLUGIN_LIBS_DIR ${PLUGIN_DIR}/libs)


##### PLUGIN SOURCES #####
list(APPEND PLUGIN_SRCS
			${PLUGIN_DIR}/src/gipDarknet.cpp
#			${PLUGIN_LIBS_DIR}/src/darknet/utils.c
#			${PLUGIN_LIBS_DIR}/src/darknet/activation_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/activations.c
#			${PLUGIN_LIBS_DIR}/src/darknet/avgpool_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/batchnorm_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/blas.c
#			${PLUGIN_LIBS_DIR}/src/darknet/box.c
#			${PLUGIN_LIBS_DIR}/src/darknet/col2im.c
#			${PLUGIN_LIBS_DIR}/src/darknet/compare.c
#			${PLUGIN_LIBS_DIR}/src/darknet/connected_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/convolutional_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/cost_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/crnn_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/crop_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/cuda.c
#			${PLUGIN_LIBS_DIR}/src/darknet/data.c
#			${PLUGIN_LIBS_DIR}/src/darknet/deconvolutional_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/detection_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/dropout_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/gemm.c
#			${PLUGIN_LIBS_DIR}/src/darknet/gru_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/im2col.c
#			${PLUGIN_LIBS_DIR}/src/darknet/image_opencv.cpp
#			${PLUGIN_LIBS_DIR}/src/darknet/image.c
#			${PLUGIN_LIBS_DIR}/src/darknet/iseg_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/l2norm_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/list.c
#			${PLUGIN_LIBS_DIR}/src/darknet/local_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/logistic_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/lstm_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/matrix.c
#			${PLUGIN_LIBS_DIR}/src/darknet/maxpool_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/network.c
#			${PLUGIN_LIBS_DIR}/src/darknet/normalization_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/option_list.c
#			${PLUGIN_LIBS_DIR}/src/darknet/parser.c
#			${PLUGIN_LIBS_DIR}/src/darknet/region_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/reorg_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/rnn_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/route_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/shortcut_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/softmax_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/tree.c
#			${PLUGIN_LIBS_DIR}/src/darknet/upsample_layer.c
#			${PLUGIN_LIBS_DIR}/src/darknet/yolo_layer.c
)


##### PLUGIN HEADERS #####
list(APPEND PLUGIN_INCLUDES
			${PLUGIN_DIR}/src
			${PLUGIN_LIBS_DIR}/include
			${PLUGIN_LIBS_DIR}/include/darknet
)


##### PLUGIN DEPENDENCY LOCATIONS (IF ANY) #####
if(WIN32)
	list(APPEND PLUGIN_LINKDIRS
				${PLUGIN_LIBS_DIR}/lib/win32
				${PLUGIN_LIBS_DIR}/bin
	)
	
	#set(PLUGIN_LIBDIRS ${PLUGIN_LIBS_DIR}/lib/win32)
	list(APPEND PLUGIN_LIBDIRS
				${PLUGIN_LIBS_DIR}/lib/win32
	)
elseif(LINUX)
	#Linux libs here
		
	add_library(darknet SHARED IMPORTED)
	set_target_properties(darknet PROPERTIES IMPORTED_LOCATION
			/home/noyan/dev/glist/glistplugins/gipDarknet/libs/lib/linux/libdarknet.so)			
	
elseif(APPLE)
	#Macos libs here
endif(WIN32)


##### PLUGIN DEPENDENCY LINKING (IF ANY) #####					
if(WIN32)
	list(APPEND PLUGIN_LINKLIBS
				#Lib names
				darknet
	)
elseif(LINUX)
	list(APPEND PLUGIN_LINKLIBS
				#Lib names
				darknet
	)
elseif(APPLE)
	list(APPEND PLUGIN_LINKLIBS
				#Lib names
	)
endif(WIN32)
