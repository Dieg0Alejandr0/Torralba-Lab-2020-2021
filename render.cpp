// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include <EGL.h>
#include <PTexLib.h>
#include <pangolin/image/image_convert.h>

#include "GLCheck.h"
#include "MirrorRenderer.h"

#include <iostream>

#include <cstdlib>
#include <random>
#include <cmath>
#include <algorithm>

/*
//Diego: We define our point-light object.
struct Light {
    vec3 position;  
  
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
	
    float constant;
    float linear;
    float quadratic;
}; 
*/


int main(int argc, char* argv[]) {

  ASSERT(argc == 5 || argc == 6, "Usage: ./ReplicaRenderer numFrames room mesh.ply /path/to/atlases [mirrorFile]");

  const std::string meshFile(argv[3]);
  const std::string atlasFolder(argv[4]);
  ASSERT(pangolin::FileExists(meshFile));
  ASSERT(pangolin::FileExists(atlasFolder));

  std::string surfaceFile;
  if (argc == 6) {
    surfaceFile = std::string(argv[5]);
    ASSERT(pangolin::FileExists(surfaceFile));
  }

  //width = 256, height = 256
  const int width = 256;
  const int height = 256;
  bool renderDepth = false;
  float depthScale = 65535.0f * 0.1f;

  //Diego: The number of frames we want
  //   and interval of camera positions
  int numFrames = atoi(argv[1]);
  int room = atoi(argv[2]);

  //Diego: We assert whether our room 
  //   argument input is correct in value
  ASSERT(0 <= room && room <= 10, "Room argument should be between 0 and 10.");

  // Setup EGL
  EGLCtx egl;

  egl.PrintInformation();
  
  if(!checkGLVersion()) {
    return 1;
  }

  //Don't draw backfaces
  const GLenum frontFace = GL_CCW;
  glFrontFace(frontFace);

  // Setup a framebuffer
  pangolin::GlTexture render(width, height);
  pangolin::GlRenderBuffer renderBuffer(width, height);
  pangolin::GlFramebuffer frameBuffer(render, renderBuffer);

  pangolin::GlTexture depthTexture(width, height, GL_R32F, false, 0, GL_RED, GL_FLOAT, 0);
  pangolin::GlFramebuffer depthFrameBuffer(depthTexture, renderBuffer);

  // load mirrors
  std::vector<MirrorSurface> mirrors;
  if (surfaceFile.length()) {
    std::ifstream file(surfaceFile);
    picojson::value json;
    picojson::parse(json, file);

    for (size_t i = 0; i < json.size(); i++) {
      mirrors.emplace_back(json[i]);
    }
    std::cout << "Loaded " << mirrors.size() << " mirrors" << std::endl;
  }

  const std::string shadir = STR(SHADER_DIR);
  MirrorRenderer mirrorRenderer(mirrors, width, height, shadir);

  // load mesh and textures
  PTexMesh ptexMesh(meshFile, atlasFolder);

  //Diego: We fix the light sources
  ptexMesh.SetExposure( 0.01 );
  ptexMesh.SetGamma( 1.697 );
  ptexMesh.SetSaturation( 1.5 );

  pangolin::ManagedImage<Eigen::Matrix<uint8_t, 3, 1>> image(width, height);
  pangolin::ManagedImage<float> depthImage(width, height);
  pangolin::ManagedImage<uint16_t> depthImageInt(width, height);

  //float camera_x = atof(argv[1]), camera_y = atof(argv[2]), camera_z = atof(argv[3]);
  //float view_x = atof(argv[4]), view_y = atof(argv[5]), view_z = atof(argv[6]);

  float camera_x, camera_y, camera_z, view_x, view_y, view_z;

  /*
  Diego: From Meshlab, I've found the following to be the 
  best points/Intervals for the camera to be in each of the listed rooms:
  
  apartment_0, king-size bed, bedroom: 
  Point: 0.827504 0.230054 3.08781
  Box (0): [-0.697871, 2.35897] [-0.283788, 2.05374] [2.29051, 3.42004] 

  apartment_0, white twin-size bed, bedroom: 
  Point: -1.10519 -4.13359 3.03902
  Box (1): [-1.92118, -0.556632]  [-5.31715, -2.99943] [2.34978, 3.34931]

  apartment_0, multicolored twin-size bed, bedroom:
  Point: 1.62124 -6.39547 2.5083
  Box (2): [0.423741, 2.35726] [-8.25771, -6.69971] [2.19416, 3.3194]

  apartment_0, living/dining room: 
  Point: -0.463154 -4.85134 0.607754
  Box (3): [-1.37427, 0.802852] [-7.4906, -1.66948] [0.00181382, 0.802461]
  
  apartment_1, dining room: 
  Point: 5.72387 4.77554 -0.0466979
  Box (4): [2.71242, 7.11203] [3.38505, 6.34683] [-0.267201, 0.755665]
  
  apartment_1, living room: 
  Point: -0.205014 3.32031 0.715108
  Box (5): [-1.0271, 1.39719] [-0.409018, 6.37123] [-0.45445, 0.881198]
  
  frl_apartment_0: 
  Point: 3.61507 -2.44515 0.80318
  Box (6): [0.243629, 4.87799] [-4.81776, -1.08331] [0.00528819, 1.21006]
  
  frl_apartment_1: 
  Point: 2.77065 3.72625 0.73041
  Box (7): [1.05569, 4.68109] [0.179999, 5.35473] [-0.214922, 1.08657]

  hotel_0: 
  Point: 0.237136 0.692084 0.62328 
  Box (8): [-2.41181, 1.83529] [-0.542538, 1.56271] [-0.0910084, 1.40267]
  
  room_0: 
  Point: 3.00275 1.16462 0.181084
  Box (9): [-0.349375, 6.39179] [-0.357708, 3.26196] [-0.0557819, 1.05521]
  
  room_1: 
  Point: -1.74929 0.210984 0.71224
  Box (10): [-3.93702, -0.0442817]  [-0.255266, -0.188404] [-0.402546, 1.12047] 
	*/

  camera_x = atof(argv[2]), camera_y = atof(argv[3]), camera_z = atof(argv[4]);

  //Diego: We set up our distributions
  std::default_random_engine generator;

  //Diego: We determine our camera position distribution based
  //   on the room argument, see the comment above about meshlab
  //   findings to understand the values used in the variables.
  float x_boundaries[11][2] = { {-0.697871, 2.35897}, {-1.92118, -0.556632},
  								{0.423741, 2.35726}, {-1.37427, 0.802852},
  								{2.71242, 7.11203}, {-1.0271, 1.39719},
  								{0.243629, 4.87799}, {1.05569, 4.68109}, 
  								{-2.41181, 1.83529}, {-0.349375, 6.39179},
  								{-3.93702, -0.0442817} };

  float y_boundaries[11][2] = { {-0.283788, 2.05374}, {-5.31715, -2.99943},
								{-8.25771, -6.69971}, {-7.4906, -1.66948},
								{3.38505, 6.34683}, {-0.409018, 6.37123},
								{-4.81776, -1.08331}, {0.179999, 5.35473}, 
								{-0.542538, 1.56271}, {-0.357708, 3.26196},
								{-0.255266, -0.188404} };

  float z_boundaries[11][2] = { {2.29051, 3.42004}, {2.34978, 3.34931}, 
								{2.19416, 3.3194}, {0.00181382, 0.802461},
								{-0.267201, 0.755665}, {-0.45445, 0.881198},
								{0.00528819, 1.21006}, {-0.214922, 1.08657}, 
								{-0.0910084, 1.40267}, {-0.0557819, 1.05521},
								{-0.402546, 1.12047} };

  std::uniform_real_distribution<float> Xcamera(x_boundaries[room][0], x_boundaries[room][1]);
  std::uniform_real_distribution<float> Ycamera(y_boundaries[room][0], y_boundaries[room][1]);	
  std::uniform_real_distribution<float> Zcamera(z_boundaries[room][0], z_boundaries[room][1]);

  std::uniform_real_distribution<float> Xview(x_boundaries[room][0], x_boundaries[room][1]);
  std::uniform_real_distribution<float> Yview(y_boundaries[room][0], y_boundaries[room][1]);
  

  for (int i = 0; i < numFrames; i++) {

  	  //Diego: We sample from our distributions
  	  camera_x = Xcamera(generator), camera_y = Ycamera(generator), camera_z = Zcamera(generator);

  	  /*
  	  std::uniform_real_distribution<float> Xview(-50 - camera_x, 50 - camera_x);
	  std::uniform_real_distribution<float> Yview(-50 - camera_y, 50 - camera_y);
	  std::normal_distribution<float> Zview(camera_z, 15);
	  */

	  float upper = z_boundaries[room][1] - camera_z, lower = camera_z - z_boundaries[room][0];
	  float standard_deviation = std::max( abs(upper), abs(lower) );

	  std::normal_distribution<float> Zview(camera_z, standard_deviation);

  	  view_x = Xview(generator), view_y = Yview(generator), view_z = Zview(generator);

	  // Setup a camera
	  pangolin::OpenGlRenderState s_cam(
	  	  
	      pangolin::ProjectionMatrixRDF_BottomLeft(
	          width,
	          height,
	          width / 2.0f,
	          width / 2.0f,
	          (width - 1.0f) / 2.0f,
	          (height - 1.0f) / 2.0f,
	          0.1f,
	          100.0f),

	      /*Diego: The arguments for model view look at RDF are as follows,
	      			(
	      			x position of camera,
	      			y position of camera,
	      			z position of camera,
					x position of viewed point,
					y position of viewed point
					z position of viewed point,
					x component of y-axis unit vector
					y component of y-axis unit vector
					z component of y-axis unit vector
	      			) */

	      pangolin::ModelViewLookAtRDF(camera_x, camera_y, camera_z, 
	      								view_x, view_y, view_z, 
	      								0, 0, 1));
	      //pangolin::ModelViewLookAtRDF(0, 0, 4, 0, 0, 0, 0, 1, 0));

	  // Start at some origin
	  //Eigen::Matrix4d T_camera_world = s_cam.GetModelViewMatrix();

	  // And move to the left
	  //Eigen::Matrix4d T_new_old = Eigen::Matrix4d::Identity();

	  //T_new_old.topRightCorner(3, 1) = Eigen::Vector3d(0.025, 0, 0);


	  // Render some frames
	  //const size_t numFrames = 1;

	  std::cout << "\rRendering frame " << i + 1 << "/" << numFrames << "... ";
	  std::cout.flush();

	  // Render
	  frameBuffer.Bind();
	  glPushAttrib(GL_VIEWPORT_BIT);
	  glViewport(0, 0, width, height);
	  glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	  glEnable(GL_CULL_FACE);

	  ptexMesh.Render(s_cam);

	  glDisable(GL_CULL_FACE);

	  glPopAttrib(); //GL_VIEWPORT_BIT
	  frameBuffer.Unbind();

	  for (size_t i = 0; i < mirrors.size(); i++) {
		MirrorSurface& mirror = mirrors[i];
		// capture reflections
		mirrorRenderer.CaptureReflection(mirror, ptexMesh, s_cam, frontFace);

		frameBuffer.Bind();
		glPushAttrib(GL_VIEWPORT_BIT);
		glViewport(0, 0, width, height);

		// render mirror
		mirrorRenderer.Render(mirror, mirrorRenderer.GetMaskTexture(i), s_cam);

		glPopAttrib(); //GL_VIEWPORT_BIT
		frameBuffer.Unbind();
	  }

	  // Download and save
	  render.Download(image.ptr, GL_RGB, GL_UNSIGNED_BYTE);

	  char filename[1000];
	  snprintf(filename, 1000, "Frame %d.jpg", i);

	  pangolin::SaveImage(
	  image.UnsafeReinterpret<uint8_t>(),
	  pangolin::PixelFormatFromString("RGB24"),
	  std::string(filename));

	  if (renderDepth) {
		// render depth
		depthFrameBuffer.Bind();
		glPushAttrib(GL_VIEWPORT_BIT);
		glViewport(0, 0, width, height);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

		glEnable(GL_CULL_FACE);

		ptexMesh.RenderDepth(s_cam, depthScale);

		glDisable(GL_CULL_FACE);

		glPopAttrib(); //GL_VIEWPORT_BIT
		depthFrameBuffer.Unbind();

		depthTexture.Download(depthImage.ptr, GL_RED, GL_FLOAT);

		// convert to 16-bit int
		for(size_t i = 0; i < depthImage.Area(); i++)
		    depthImageInt[i] = static_cast<uint16_t>(depthImage[i] + 0.5f);

		snprintf(filename, 1000, "Depth Frame %d.jpg", i);
		pangolin::SaveImage(
		    depthImageInt.UnsafeReinterpret<uint8_t>(),
		    pangolin::PixelFormatFromString("GRAY16LE"),
		    std::string(filename), true, 34.0f);
	    }
	}

  // Move the camera
  //T_camera_world = T_camera_world * T_new_old.inverse();

  //s_cam.GetModelViewMatrix() = T_camera_world;
  
  std::cout << "\rRendering frame " << numFrames << "/" << numFrames << "... done \n";
  //std::cout << "Camera at (" << camera_x << ", " << camera_y << ", " << camera_z << "), Looking at (" << view_x << ", " << view_y << ", " << view_z << ") \n" << std::endl;

  return 0;

  }
