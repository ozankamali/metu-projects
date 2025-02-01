#ifndef _SCENE_H_
#define _SCENE_H_
#include <vector>
#include <iostream>

#include "Helpers.h"
#include "Vec3.h"
#include "Vec4.h"
#include "Color.h"
#include "Rotation.h"
#include "Scaling.h"
#include "Translation.h"
#include "Camera.h"
#include "Mesh.h"

class Scene
{
public:
	Color backgroundColor;
	bool cullingEnabled;

	std::vector<std::vector<Color> > image;
	std::vector<std::vector<float>> depthBuffer;
	std::vector<Camera *> cameras;
	std::vector<Vec3 *> vertices;
	std::vector<Color *> colorsOfVertices;
	std::vector<Scaling *> scalings;
	std::vector<Rotation *> rotations;
	std::vector<Translation *> translations;
	std::vector<Mesh *> meshes;

	Scene(const char *xmlPath);

	void initializeImage(Camera *camera);
	int makeBetweenZeroAnd255(double value);
	void writeImageToPPMFile(Camera *camera);
	void convertPPMToPNG(std::string ppmFileName, int osType);
	
	Matrix4 getTranslationToOrigin(const double &tx, const double &ty, const double &tz);
	Matrix4 getTranslationMatrix(int &translationId);
	Matrix4 getScalingMatrix(int &scalingId);
	Matrix4 getRotationMatrix(int &rotationId);
	std::vector<Matrix4> modelingTransformation();
	Matrix4 cameraTransformation(const auto& camera);
	Matrix4 projectionTransformation(const auto& camera);
	Matrix4 viewportTransformation(const auto& camera);
	bool isBackfacing(Vec4 &v1, Vec4 &v2, Vec4 &v3);
	bool visible(double den, double num, double &tE, double &tL);
	bool liangBarskyClipping(std::vector<Vec4> &line);
	void drawPixel(int x, int y, double z, const Color &color, const auto& camera);
	void rasterizeLine(const Vec3 &v1, const Vec3 &v0, const auto& camera);
	double computeEdge(const Vec3 &v0, const Vec3 &v1, const double &x, const double &y);
	void rasterizeTriangle(const Vec3 &v1, const Vec3 &v2, const Vec3 &v3, const auto& camera);
	void forwardRenderingPipeline(Camera *camera);
};	

#endif
