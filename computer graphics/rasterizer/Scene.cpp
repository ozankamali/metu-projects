#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <limits> 

#include "tinyxml2.h"
#include "Triangle.h"
#include "Helpers.h"
#include "Scene.h"

using namespace tinyxml2;
using namespace std;

/*
	Parses XML file
*/
Scene::Scene(const char *xmlPath)
{
	const char *str;
	XMLDocument xmlDoc;
	XMLElement *xmlElement;

	xmlDoc.LoadFile(xmlPath);

	XMLNode *rootNode = xmlDoc.FirstChild();

	// read background color
	xmlElement = rootNode->FirstChildElement("BackgroundColor");
	str = xmlElement->GetText();
	sscanf(str, "%lf %lf %lf", &backgroundColor.r, &backgroundColor.g, &backgroundColor.b);

	// read culling
	xmlElement = rootNode->FirstChildElement("Culling");
	if (xmlElement != NULL)
	{
		str = xmlElement->GetText();

		if (strcmp(str, "enabled") == 0)
		{
			this->cullingEnabled = true;
		}
		else
		{
			this->cullingEnabled = false;
		}
	}

	// read cameras
	xmlElement = rootNode->FirstChildElement("Cameras");
	XMLElement *camElement = xmlElement->FirstChildElement("Camera");
	XMLElement *camFieldElement;
	while (camElement != NULL)
	{
		Camera *camera = new Camera();

		camElement->QueryIntAttribute("id", &camera->cameraId);

		// read projection type
		str = camElement->Attribute("type");

		if (strcmp(str, "orthographic") == 0)
		{
			camera->projectionType = ORTOGRAPHIC_PROJECTION;
		}
		else
		{
			camera->projectionType = PERSPECTIVE_PROJECTION;
		}

		camFieldElement = camElement->FirstChildElement("Position");
		str = camFieldElement->GetText();
		sscanf(str, "%lf %lf %lf", &camera->position.x, &camera->position.y, &camera->position.z);

		camFieldElement = camElement->FirstChildElement("Gaze");
		str = camFieldElement->GetText();
		sscanf(str, "%lf %lf %lf", &camera->gaze.x, &camera->gaze.y, &camera->gaze.z);

		camFieldElement = camElement->FirstChildElement("Up");
		str = camFieldElement->GetText();
		sscanf(str, "%lf %lf %lf", &camera->v.x, &camera->v.y, &camera->v.z);

		camera->gaze = normalizeVec3(camera->gaze);
		camera->u = crossProductVec3(camera->gaze, camera->v);
		camera->u = normalizeVec3(camera->u);

		camera->w = inverseVec3(camera->gaze);
		camera->v = crossProductVec3(camera->u, camera->gaze);
		camera->v = normalizeVec3(camera->v);

		camFieldElement = camElement->FirstChildElement("ImagePlane");
		str = camFieldElement->GetText();
		sscanf(str, "%lf %lf %lf %lf %lf %lf %d %d",
			   &camera->left, &camera->right, &camera->bottom, &camera->top,
			   &camera->near, &camera->far, &camera->horRes, &camera->verRes);

		camFieldElement = camElement->FirstChildElement("OutputName");
		str = camFieldElement->GetText();
		camera->outputFilename = string(str);

		this->cameras.push_back(camera);

		camElement = camElement->NextSiblingElement("Camera");
	}

	// read vertices
	xmlElement = rootNode->FirstChildElement("Vertices");
	XMLElement *vertexElement = xmlElement->FirstChildElement("Vertex");
	int vertexId = 1;

	while (vertexElement != NULL)
	{
		Vec3 *vertex = new Vec3();
		Color *color = new Color();

		vertex->colorId = vertexId;

		str = vertexElement->Attribute("position");
		sscanf(str, "%lf %lf %lf", &vertex->x, &vertex->y, &vertex->z);

		str = vertexElement->Attribute("color");
		sscanf(str, "%lf %lf %lf", &color->r, &color->g, &color->b);

		this->vertices.push_back(vertex);
		this->colorsOfVertices.push_back(color);

		vertexElement = vertexElement->NextSiblingElement("Vertex");

		vertexId++;
	}

	// read translations
	xmlElement = rootNode->FirstChildElement("Translations");
	XMLElement *translationElement = xmlElement->FirstChildElement("Translation");
	while (translationElement != NULL)
	{
		Translation *translation = new Translation();

		translationElement->QueryIntAttribute("id", &translation->translationId);

		str = translationElement->Attribute("value");
		sscanf(str, "%lf %lf %lf", &translation->tx, &translation->ty, &translation->tz);

		this->translations.push_back(translation);

		translationElement = translationElement->NextSiblingElement("Translation");
	}

	// read scalings
	xmlElement = rootNode->FirstChildElement("Scalings");
	XMLElement *scalingElement = xmlElement->FirstChildElement("Scaling");
	while (scalingElement != NULL)
	{
		Scaling *scaling = new Scaling();

		scalingElement->QueryIntAttribute("id", &scaling->scalingId);
		str = scalingElement->Attribute("value");
		sscanf(str, "%lf %lf %lf", &scaling->sx, &scaling->sy, &scaling->sz);

		this->scalings.push_back(scaling);

		scalingElement = scalingElement->NextSiblingElement("Scaling");
	}

	// read rotations
	xmlElement = rootNode->FirstChildElement("Rotations");
	XMLElement *rotationElement = xmlElement->FirstChildElement("Rotation");
	while (rotationElement != NULL)
	{
		Rotation *rotation = new Rotation();

		rotationElement->QueryIntAttribute("id", &rotation->rotationId);
		str = rotationElement->Attribute("value");
		sscanf(str, "%lf %lf %lf %lf", &rotation->angle, &rotation->ux, &rotation->uy, &rotation->uz);

		this->rotations.push_back(rotation);

		rotationElement = rotationElement->NextSiblingElement("Rotation");
	}

	// read meshes
	xmlElement = rootNode->FirstChildElement("Meshes");

	XMLElement *meshElement = xmlElement->FirstChildElement("Mesh");
	while (meshElement != NULL)
	{
		Mesh *mesh = new Mesh();

		meshElement->QueryIntAttribute("id", &mesh->meshId);

		// read projection type
		str = meshElement->Attribute("type");

		if (strcmp(str, "wireframe") == 0)
		{
			mesh->type = WIREFRAME_MESH;
		}
		else
		{
			mesh->type = SOLID_MESH;
		}

		// read mesh transformations
		XMLElement *meshTransformationsElement = meshElement->FirstChildElement("Transformations");
		XMLElement *meshTransformationElement = meshTransformationsElement->FirstChildElement("Transformation");

		while (meshTransformationElement != NULL)
		{
			char transformationType;
			int transformationId;

			str = meshTransformationElement->GetText();
			sscanf(str, "%c %d", &transformationType, &transformationId);

			mesh->transformationTypes.push_back(transformationType);
			mesh->transformationIds.push_back(transformationId);

			meshTransformationElement = meshTransformationElement->NextSiblingElement("Transformation");
		}

		mesh->numberOfTransformations = mesh->transformationIds.size();

		// read mesh faces
		char *row;
		char *cloneStr;
		int v1, v2, v3;
		XMLElement *meshFacesElement = meshElement->FirstChildElement("Faces");
		str = meshFacesElement->GetText();
		cloneStr = strdup(str);

		row = strtok(cloneStr, "\n");
		while (row != NULL)
		{
			int result = sscanf(row, "%d %d %d", &v1, &v2, &v3);

			if (result != EOF)
			{
				mesh->triangles.push_back(Triangle(v1, v2, v3));
			}
			row = strtok(NULL, "\n");
		}
		mesh->numberOfTriangles = mesh->triangles.size();
		this->meshes.push_back(mesh);

		meshElement = meshElement->NextSiblingElement("Mesh");
	}
}

/*
	Initializes image with background color
*/
// void Scene::initializeImage(Camera *camera)
// {
// 	if (this->image.empty())
// 	{
// 		for (int i = 0; i < camera->horRes; i++)
// 		{
// 			vector<Color> rowOfColors;

// 			for (int j = 0; j < camera->verRes; j++)
// 			{
// 				rowOfColors.push_back(this->backgroundColor);
// 			}

// 			this->image.push_back(rowOfColors);
// 		}
// 	}
// 	else
// 	{
// 		for (int i = 0; i < camera->horRes; i++)
// 		{
// 			for (int j = 0; j < camera->verRes; j++)
// 			{
// 				this->image[i][j].r = this->backgroundColor.r;
// 				this->image[i][j].g = this->backgroundColor.g;
// 				this->image[i][j].b = this->backgroundColor.b;
// 			}
// 		}
// 	}
// }

void Scene::initializeImage(Camera *camera)
{
    this->image.clear();
    this->depthBuffer.clear();

    for (int i = 0; i < camera->horRes; i++)
    {
        std::vector<Color> rowOfColors;
        std::vector<float> rowOfDepths;

        for (int j = 0; j < camera->verRes; j++)
        {
            rowOfColors.push_back(this->backgroundColor);
            rowOfDepths.push_back(1.0f);
        }

        this->image.push_back(rowOfColors);
        this->depthBuffer.push_back(rowOfDepths);
    }
}


/*
	If given value is less than 0, converts value to 0.
	If given value is more than 255, converts value to 255.
	Otherwise returns value itself.
*/
int Scene::makeBetweenZeroAnd255(double value)
{
	if (value >= 255.0)
		return 255;
	if (value <= 0.0)
		return 0;
	return (int)(value);
}


/*
	Writes contents of image (Color**) into a PPM file.
*/
void Scene::writeImageToPPMFile(Camera *camera)
{
	ofstream fout;

	fout.open(camera->outputFilename.c_str());

	fout << "P3" << endl;
	fout << "# " << camera->outputFilename << endl;
	fout << camera->horRes << " " << camera->verRes << endl;
	fout << "255" << endl;

	for (int j = camera->verRes - 1; j >= 0; j--)
	{
		for (int i = 0; i < camera->horRes; i++)
		{
			fout << makeBetweenZeroAnd255(this->image[i][j].r) << " "
				 << makeBetweenZeroAnd255(this->image[i][j].g) << " "
				 << makeBetweenZeroAnd255(this->image[i][j].b) << " ";
		}
		fout << endl;
	}
	fout.close();
}

/*
	Converts PPM image in given path to PNG file, by calling ImageMagick's 'convert' command.
	os_type == 1 		-> Ubuntu
	os_type == 2 		-> Windows
	os_type == other	-> No conversion
*/
void Scene::convertPPMToPNG(string ppmFileName, int osType)
{
    string command;
    string outputFolder = "my_outputs/";
    system(("mkdir -p " + outputFolder).c_str()); // Linux/Mac: create folder

    // Add the output folder path to the output filename
    string outputFileName = outputFolder + ppmFileName + ".png";

    // Call command on Ubuntu/Linux/Mac
    if (osType == 1)
    {
        command = "./magick " + ppmFileName + " " + outputFileName;
        system(command.c_str());
    }
    // Call command on Windows
    else if (osType == 2)
    {
        command = "magick " + ppmFileName + " " + outputFileName;
        system(command.c_str());
    }
    // Default action - do nothing
    else
    {
    }
}


















Matrix4 Scene::getTranslationToOrigin(const double &tx, const double &ty, const double &tz) {
    Matrix4 translationMatrix = getIdentityMatrix();
    translationMatrix.set(0,3, -tx);
    translationMatrix.set(1,3, -ty);
    translationMatrix.set(2,3, -tz);

    return translationMatrix;
}

Matrix4 Scene::getTranslationMatrix(int &translationId) {
    const auto& translation = this->translations[translationId - 1];
    double tx = translation->tx;
    double ty = translation->ty;
    double tz = translation->tz;

    Matrix4 translationMatrix = getIdentityMatrix();
    translationMatrix.set(0,3, tx);
    translationMatrix.set(1,3, ty);
    translationMatrix.set(2,3, tz);

    return translationMatrix;
}

Matrix4 Scene::getScalingMatrix(int &scalingId) {
    const auto& scaling = this->scalings[scalingId - 1];
    double sx = scaling->sx;
    double sy = scaling->sy;
    double sz = scaling->sz;

    Matrix4 scalingMatrix = getIdentityMatrix();
    scalingMatrix.set(0,0, sx);
    scalingMatrix.set(1,1, sy);
    scalingMatrix.set(2,2, sz);

    return scalingMatrix;
}


Matrix4 Scene::getRotationMatrix(int &rotationId) {
    const auto& rotation = this->rotations[rotationId - 1];
    
	Vec3 v;
	Vec3 w;
	double ux = rotation->ux;
    double uy = rotation->uy;
	double uz = rotation->uz;
    Vec3 u = Vec3(ux, uy, uz);

	double angle = rotation->angle * M_PI / 180.0; // IN RADIANS
	
	double min_component = std::min({std::abs(ux), std::abs(uy), std::abs(uz)});
	if (min_component == std::abs(ux)){
		v = Vec3(0, -uz, uy);
	}
	else if (min_component == std::abs(uy)){
		v = Vec3(-uz, 0, ux);
	}
	else if (min_component == std::abs(uz)){
		v = Vec3(-uy, ux, 0);
	}
	
	w = crossProductVec3(u, v);
	v = normalizeVec3(v);
	w = normalizeVec3(w);
	
	double vx = v.x;
    double vy = v.y;
	double vz = v.z;

	double wx = w.x;
    double wy = w.y;
	double wz = w.z;


    Matrix4 rotationMatrix = getIdentityMatrix();
	Matrix4 M = getIdentityMatrix(); 
	Matrix4 M_inverse = getIdentityMatrix();
	
	rotationMatrix.set(1,1, std::cos(angle));
	rotationMatrix.set(1,2, (-1) * std::sin(angle));
	rotationMatrix.set(2,1, std::sin(angle));
	rotationMatrix.set(2,2, std::cos(angle));

	M.set(0,0, ux);
	M.set(0,1, uy);
	M.set(0,2, uz);
	
	M.set(1,0, vx);
	M.set(1,1, vy);
	M.set(1,2, vz);

	M.set(2,0, wx);
	M.set(2,1, wy);
	M.set(2,2, wz);
	
	M_inverse.set(0,0, ux);
	M_inverse.set(0,1, vx);
	M_inverse.set(0,2, wx);
	
	M_inverse.set(1,0, uy);
	M_inverse.set(1,1, vy);
	M_inverse.set(1,2, wy);

	M_inverse.set(2,0, uz);
	M_inverse.set(2,1, vz);
	M_inverse.set(2,2, wz);
	// T_-origin * M_-1 * R_x(angle) * M * T_origin
	const Matrix4 Rx_M = multiplyMatrixWithMatrix(rotationMatrix, M);
	const Matrix4 M_1_Rx_M = multiplyMatrixWithMatrix(M_inverse, Rx_M);
    return M_1_Rx_M;
}






// RETURN: VECTOR OF Composite Matrix for every mesh
std::vector<Matrix4> Scene::modelingTransformation() {
	std::vector<Matrix4> transformationMatrices;
    for (const auto& mesh : this->meshes) {
    	Matrix4 transformationMatrix = getIdentityMatrix(); 
        for (int i = mesh->numberOfTransformations - 1; i >= 0; i--) {
            char type = mesh->transformationTypes[i];
            int id = mesh->transformationIds[i];
            if (type == 't') {
                transformationMatrix = multiplyMatrixWithMatrix(transformationMatrix, getTranslationMatrix(id));
            } 
			else if (type == 'r') {
                transformationMatrix = multiplyMatrixWithMatrix(transformationMatrix, getRotationMatrix(id));
            } 
			else if (type == 's') {
                transformationMatrix = multiplyMatrixWithMatrix(transformationMatrix, getScalingMatrix(id));
            }
        }
		transformationMatrices.push_back(transformationMatrix);
	}
	return transformationMatrices;
}

Matrix4 Scene::cameraTransformation(const auto& camera){
	Matrix4 rotationM = getIdentityMatrix();
	rotationM.set(0,0, camera->u.x);
	rotationM.set(0,1, camera->u.y);
	rotationM.set(0,2, camera->u.z);

	rotationM.set(1,0, camera->v.x);
	rotationM.set(1,1, camera->v.y);
	rotationM.set(1,2, camera->v.z);

	rotationM.set(2,0, camera->w.x);
	rotationM.set(2,1, camera->w.y);
	rotationM.set(2,2, camera->w.z);

	Matrix4 cameraM = multiplyMatrixWithMatrix(rotationM, getTranslationToOrigin(camera->position.x, camera->position.y, camera->position.z));
	return cameraM;
}


Matrix4 Scene::projectionTransformation(const auto& camera){

	double l = camera->left;
	double r = camera->right;
	double b = camera->bottom;
	double t = camera->top;
	double n = camera->near;
	double f = camera->far;
	
	Matrix4 ortMatrix = getIdentityMatrix();
	ortMatrix.set(0,0, 2 / (r-l));
	ortMatrix.set(0,3, -(r+l) / (r-l));
	ortMatrix.set(1,1, 2 / (t-b));
	ortMatrix.set(1,3, -(t+b) / (t-b));
	ortMatrix.set(2,2, -2 / (f-n));
	ortMatrix.set(2,3, -(f+n) / (f-n));
	
	if (camera->projectionType == 0) { // orthographic projection
		return ortMatrix;
	} 

	else { // perspective projection
		Matrix4 perspectiveMatrix = getIdentityMatrix();
		perspectiveMatrix.set(0,0, (2*n) / (r-l));
		perspectiveMatrix.set(0,2, (r+l) / (r-l));
		perspectiveMatrix.set(1,1, (2*n) / (t-b));
		perspectiveMatrix.set(1,2, (t+b) / (t-b));
		perspectiveMatrix.set(2,2, -(f+n) / (f-n));
		perspectiveMatrix.set(2,3, -2*(f*n) / (f-n));
		perspectiveMatrix.set(3,2, -1.0);
		perspectiveMatrix.set(3,3, 0.0);
		return perspectiveMatrix;	
	}

}

Matrix4 Scene::viewportTransformation(const auto& camera){

	double nx = camera->horRes;
	double ny = camera->verRes;

	Matrix4 viewportMatrix = getIdentityMatrix();
	viewportMatrix.set(0,0, nx / 2.0);
	viewportMatrix.set(0,3, (nx - 1) / 2.0);
	viewportMatrix.set(1,1, ny / 2.0);
	viewportMatrix.set(1,3, (ny - 1) / 2.0);
	viewportMatrix.set(2,2, 0.5);
	viewportMatrix.set(2,3, 0.5);
	return viewportMatrix;
}

bool Scene::isBackfacing(Vec4 &v1, Vec4 &v2, Vec4 &v3){
	Vec3 e21 = subtractVec3(Vec4toVec3(v2), Vec4toVec3(v1));
	Vec3 e31 = subtractVec3(Vec4toVec3(v3), Vec4toVec3(v1));
	
	Vec3 n = crossProductVec3(e21, e31);
	n = normalizeVec3(n);
	double cos = dotProductVec3(n, Vec4toVec3(v1));

	return (cos < 0);
}

bool Scene::visible(double den, double num, double &tE, double &tL){
	double t;
	if (den > 0) {
		t = num / den;
		if (t > tL) {
			return false;
		}
		if (t > tE) {
			tE = t;
		}
	}
	else if (den < 0) {
		t = num / den;
		if (t < tE) {
			return false;
		}
		if (t < tL) {
			tL = t;
		}
	}
	else {
		if (num > 0) {
			return false;
		}
	} 
	return true;
}

bool Scene::liangBarskyClipping(std::vector<Vec4> &line)
{
	Vec4 v1 = line[0];
	Vec4 v0 = line[1];
	double xMin = -1.0; 
	double yMin = -1.0; 
	double zMin = -1.0;
	double xMax = 1.0; 
	double yMax = 1.0; 
	double zMax = 1.0;

	double dx = v1.x-v0.x;
	double dy = v1.y-v0.y;
	double dz = v1.z-v0.z;

	double tE = 0.0;
	double tL = 1.0;

	bool visible_ = false;
	
	if (visible(dx, xMin - v0.x, tE, tL)){
		if (visible(-dx, v0.x - xMax, tE, tL)){
			if (visible(dy, yMin - v0.y, tE, tL)){
				if (visible(-dy, v0.y - yMax, tE, tL)){
					if (visible(dz, zMin - v0.z, tE, tL)){
						if (visible(-dz, v0.z - zMax, tE, tL)){
							visible_ = true;
							if (tL < 1){
								v1.x = v0.x + dx*tL;
								v1.y = v0.y + dy*tL;
								v1.z = v0.z + dz*tL;
							}
							if (tE > 0){
								v0.x = v0.x + dx*tE;
								v0.y = v0.y + dy*tE;
								v0.z = v0.z + dz*tE;
							}
						} 
					}
				} 
			}
		}
	} 
	line[0] = v1;
	line[1] = v0;
	return visible_;	
}


// void Scene::drawPixel(int x, int y, const Color &color, const auto& camera) {
//     if (x >= 0 && x < camera->horRes && y >= 0 && y < camera->verRes) {
//         this->image[x][y] = color;
//     }
// }

void Scene::drawPixel(int x, int y, double z, const Color &color, const auto &camera)
{
    if (x >= 0 && x < camera->horRes && y >= 0 && y < camera->verRes)
    {
        if (z < this->depthBuffer[x][y])
        {
            this->depthBuffer[x][y] = z; 
            this->image[x][y] = color;  
        }
    }
}

void Scene::rasterizeLine(const Vec3 &v0, const Vec3 &v1, const auto& camera) {

    int x0 = static_cast<int>(v0.x);
    int y0 = static_cast<int>(v0.y);
    int x1 = static_cast<int>(v1.x);
    int y1 = static_cast<int>(v1.y);
	
	double z0 = v0.z;
	double z1 = v1.z;

	int dx = x1 - x0;
    int dy = y1 - y0;
	Color c;
	Color dc;
    int step = 1; 
	
	Color c0 = *(this->colorsOfVertices[v0.colorId]);
	Color c1 = *(this->colorsOfVertices[v1.colorId]);


    if (abs(dy) <= abs(dx)) {  // 0 <= m <= 1
        if (x1 < x0){
			std::swap(x1, x0);
			std::swap(y1, y0); 
			std::swap(c1, c0);
			std::swap(z1, z0);
		}
		if (y1 < y0){
			step = -1;
		}
		
        int y = y0;
		int d = 2 * (y0 - y1) + step *(x1 - x0);
		c = c0;
		dc = Color((c1.r - c0.r) / (x1 - x0), (c1.g - c0.g) / (x1 - x0), (c1.b - c0.b) / (x1 - x0));  
        for (int x = x0; x <= x1; x++) {
		
            drawPixel(x, y, z0, Color(static_cast<int>(c.r), static_cast<int>(c.g), static_cast<int>(c.b)), camera);  
            

			if (d*step < 0) {  // NE 
                y += step;
                d += 2 * ((y0 - y1) + step * (x1 - x0));
            } else {  // E 
                d += 2 * (y0 - y1);
            }
			c.r += dc.r;
			c.g += dc.g;
			c.b += dc.b;
        }
    } 
	else {  // m > 1
        if (y1 < y0){
			std::swap(x1, x0);
			std::swap(y1, y0);
			std::swap(c1, c0); 
			std::swap(z1, z0);
		}
		if (x1 < x0){
			step = -1;
		}
		
        int x = x0;
		int d = 2 * (x1 - x0) + step * (y0 - y1);
		c = c0;
		dc = Color((c1.r - c0.r) / (y1 - y0), (c1.g - c0.g) / (y1 - y0), (c1.b - c0.b) / (y1 - y0));   
        for (int y = y0; y <= y1; y++) {
            drawPixel(x, y, z0, Color(static_cast<int>(c.r), static_cast<int>(c.g), static_cast<int>(c.b)), camera);  
			
            if (d*step > 0) {  
                x += step;
                d += 2 * ((x1 - x0) + step * (y0 - y1));
            } else {  
                d += 2 * (x1 - x0);
            }
			c.r += dc.r;
			c.g += dc.g;
			c.b += dc.b;
        }
    }
}


double Scene::computeEdge(const Vec3 &v0, const Vec3 &v1, const double &x, const double &y){
	return ((v1.x - v0.x) * (y - v0.y) - (v1.y - v0.y) * (x - v0.x));
}

void Scene::rasterizeTriangle(const Vec3 &v1, const Vec3 &v2, const Vec3 &v3, const auto &camera) {
    int xMin = static_cast<int>(std::max(0.0, std::min({v1.x, v2.x, v3.x})));
    int yMin = static_cast<int>(std::max(0.0, std::min({v1.y, v2.y, v3.y})));
    int xMax = static_cast<int>(std::min(camera->horRes - 1.0, std::max({v1.x, v2.x, v3.x})));
    int yMax = static_cast<int>(std::min(camera->verRes - 1.0, std::max({v1.y, v2.y, v3.y})));

    Color c1 = *(this->colorsOfVertices[v1.colorId]);
    Color c2 = *(this->colorsOfVertices[v2.colorId]);
    Color c3 = *(this->colorsOfVertices[v3.colorId]);
    Color c;

	double z1 = v1.z;
	double z2 = v2.z;
	double z3 = v3.z;

    double totalArea = std::abs(computeEdge(v1, v2, v3.x, v3.y));

    for (int y = yMin; y <= yMax; y++) {
        for (int x = xMin; x <= xMax; x++) {
            double A = computeEdge(v1, v2, x, y) / totalArea;
            double B = computeEdge(v2, v3, x, y) / totalArea;
            double C = computeEdge(v3, v1, x, y) / totalArea;
			float z_prime = z3*A+z1*B+z2*C;
            if ((A >= 0 && B >= 0 && C >= 0)) {
                c.r = static_cast<int>((c3.r * A) + (c1.r * B) + (c2.r * C));
				c.g = static_cast<int>((c3.g * A) + (c1.g * B) + (c2.g * C));
				c.b = static_cast<int>((c3.b * A) + (c1.b * B) + (c2.b * C));
				// z interpolation
                drawPixel(x, y, z_prime, c, camera);
            }
        }
    }
}

void Scene::forwardRenderingPipeline(Camera *camera)
{
	
	// GET ALL MATRICES
	std::vector<Matrix4> modelMatrices = this->modelingTransformation();
	const Matrix4 cameraMatrix = this->cameraTransformation(camera);
	const Matrix4 projectionMatrix = this->projectionTransformation(camera);
	const Matrix4 viewportMatrix = this->viewportTransformation(camera);


	for (size_t i = 0; i < this->meshes.size(); ++i) {		
		const auto& mesh = this->meshes[i];
		// M_proj * M_cam * M_model 
		const Matrix4 M_cam_M_model = multiplyMatrixWithMatrix(cameraMatrix, modelMatrices[i]);
		const Matrix4 M_proj_M_cam_M_model = multiplyMatrixWithMatrix(projectionMatrix, M_cam_M_model);

        for (const auto& triangle : mesh->triangles) {
			// Mproj * Mcam * Mmodel * v
			auto v1_ = this->vertices[triangle.vertexIds[0] - 1];
			auto v2_ = this->vertices[triangle.vertexIds[1] - 1];
			auto v3_ = this->vertices[triangle.vertexIds[2] - 1];
			auto c1_ = this->colorsOfVertices[triangle.vertexIds[0] - 1];
			auto c2_ = this->colorsOfVertices[triangle.vertexIds[1] - 1];
			auto c3_ = this->colorsOfVertices[triangle.vertexIds[2] - 1];
		

			Vec4 v1 = Vec4(v1_->x, v1_->y, v1_->z, double(1.0), triangle.vertexIds[0] - 1);
			Vec4 v2 = Vec4(v2_->x, v2_->y, v2_->z, double(1.0), triangle.vertexIds[1] - 1);
			Vec4 v3 = Vec4(v3_->x, v3_->y, v3_->z, double(1.0), triangle.vertexIds[2] - 1);
			
			
			v1 = multiplyMatrixWithVec4(M_proj_M_cam_M_model, v1);
            v2 = multiplyMatrixWithVec4(M_proj_M_cam_M_model, v2);
            v3 = multiplyMatrixWithVec4(M_proj_M_cam_M_model, v3);

			
			// backface culling 
			if (this->cullingEnabled && isBackfacing(v1, v2, v3)){
				continue;
			}

			// WIREFRAME
			if(mesh->type == 0){	

				std::vector<Vec4> line21;
				std::vector<Vec4> line31;
				std::vector<Vec4> line32;
				
				// clipping (for wireframe mode only)
				line21.push_back(v2);
				line21.push_back(v1);

				line31.push_back(v3);
				line31.push_back(v1);

				line32.push_back(v3);
				line32.push_back(v2);
				
				//clipping 
				// bool line21ClippingVisible = liangBarskyClipping(line21);
				// bool line31ClippingVisible = liangBarskyClipping(line31);
				// bool line32ClippingVisible = liangBarskyClipping(line32);

				//perspective divide	
				// v2'
				line21[0].x /= line21[0].w; 
				line21[0].y /= line21[0].w; 
				line21[0].z /= line21[0].w; 
				line21[0].w = 1.0; 
				// v1'
				line21[1].x /= line21[1].w; 
				line21[1].y /= line21[1].w; 
				line21[1].z /= line21[1].w; 
				line21[1].w = 1.0; 
				// v3'
				line31[0].x /= line31[0].w; 
				line31[0].y /= line31[0].w; 
				line31[0].z /= line31[0].w; 
				line31[0].w = 1.0; 
				// v1''
				line31[1].x /= line31[1].w; 
				line31[1].y /= line31[1].w; 
				line31[1].z /= line31[1].w; 
				line31[1].w = 1.0; 
				// v3''
				line32[0].x /= line32[0].w; 
				line32[0].y /= line32[0].w; 
				line32[0].z /= line32[0].w; 
				line32[0].w = 1.0; 
				// v2''
				line32[1].x /= line32[1].w; 
				line32[1].y /= line32[1].w; 
				line32[1].z /= line32[1].w; 
				line32[1].w = 1.0; 

				// // Mvp * Mproj * Mcam * Mmodel * v
				line21[0] = multiplyMatrixWithVec4(viewportMatrix, line21[0]);
				line21[1] = multiplyMatrixWithVec4(viewportMatrix, line21[1]);
				line31[0] = multiplyMatrixWithVec4(viewportMatrix, line31[0]);
				line31[1] = multiplyMatrixWithVec4(viewportMatrix, line31[1]);
				line32[0] = multiplyMatrixWithVec4(viewportMatrix, line32[0]);
				line32[1] = multiplyMatrixWithVec4(viewportMatrix, line32[1]);

				// if(line21ClippingVisible){
				// }
				// if(line31ClippingVisible){
				// }
				// if(line32ClippingVisible){
				// }
			
				rasterizeLine(Vec4toVec3(line21[0]), Vec4toVec3(line21[1]), camera); // v2, v1
				rasterizeLine(Vec4toVec3(line31[0]), Vec4toVec3(line31[1]), camera); // v3, v1'
				rasterizeLine(Vec4toVec3(line32[0]), Vec4toVec3(line32[1]), camera); // v3', v2'
			}



			// SOLID
			else{
				//perspective d
				v1.x /= v1.w;
				v1.y /= v1.w;
				v1.z /= v1.w;
				v1.w = 1.0;

				v2.x /= v2.w;
				v2.y /= v2.w;
				v2.z /= v2.w;
				v2.w = 1.0;

				v3.x /= v3.w;
				v3.y /= v3.w;
				v3.z /= v3.w;
				v3.w = 1.0;

				//viewport 
				v1 = multiplyMatrixWithVec4(viewportMatrix, v1);
				v2 = multiplyMatrixWithVec4(viewportMatrix, v2);
				v3 = multiplyMatrixWithVec4(viewportMatrix, v3); 
				
				//rasterize
				rasterizeTriangle(Vec4toVec3(v1), Vec4toVec3(v2), Vec4toVec3(v3), camera);
			}
		}
	}	
}
