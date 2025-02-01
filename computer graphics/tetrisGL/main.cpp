#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#define _USE_MATH_DEFINES
#include <math.h>
#include <GL/glew.h>
//#include <OpenGL/gl3.h>   // The GL Header File
#include <GLFW/glfw3.h> // The GLFW header
#include <glm/glm.hpp> // GL Math library header
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp> 
#include <ft2build.h>
#include <algorithm> // for vector remove
#include FT_FREETYPE_H

#define BUFFER_OFFSET(i) ((char*)NULL + (i))

using namespace std;

GLuint gProgram[4];
int gWidth = 600, gHeight = 1000;
GLuint gVertexAttribBuffer, gTextVBO, gIndexBuffer;
GLuint gTex2D;
int gVertexDataSizeInBytes, gNormalDataSizeInBytes;
int gTriangleIndexDataSizeInBytes, gLineIndexDataSizeInBytes;

GLint modelingMatrixLoc[3];
GLint viewingMatrixLoc[3];
GLint projectionMatrixLoc[3];
GLint eyePosLoc[3];
GLint lightPosLoc[3];
GLint kdLoc[3];

glm::mat4 projectionMatrix;
glm::mat4 viewingMatrix;
glm::mat4 modelingMatrix = glm::translate(glm::mat4(1.f), glm::vec3(-1.5f, 7.f, -1.5f));
glm::mat4 gridModel = modelingMatrix;
glm::mat4 baseModelMatrix = modelingMatrix;


glm::vec3 eyePos = glm::vec3(0, 5, 26.f);
glm::vec3 lightPos = glm::vec3(0, 5, 7);

glm::vec3 kdGround(0.4, 0.1, 0.1); 
glm::vec3 kdCubes(0.0, 0.76, 0.2);


float xLimit = 0.f;
float zLimit = 0.f;
float yLimit = 0.f;
float ground_coord = -12.0f;

int k = 9000;
int mod = 45;

int activeProgramIndex = 0;
float falling_speed = 1.f;
int eye_k = 0;
glm::vec3 eye_Left = glm::vec3(-26.f, 5.f, 0.f);
glm::vec3 eye_Right = glm::vec3(26.f, 5.f, 0.f);
glm::vec3 eye_Front = glm::vec3(0.f, 5.f, 26.f);
glm::vec3 eye_Back = glm::vec3(0.f, 5.f, -26.f);

glm::vec3 light_Left = glm::vec3(-7, 5, 0) ;
glm::vec3 light_Right = glm::vec3(7, 5, 0) ;
glm::vec3 light_Front = glm::vec3(0, 5, 7) ;
glm::vec3 light_Back = glm::vec3(0, 5, -7) ;

bool isTransitioning = false;

bool isH = false;
bool isK = false;
bool isA = false;
bool isD = false;
bool isS = false;
bool isW = false;

float transitionSpeed = 0.f;
glm::vec3 transitionVector;

glm::vec3 startEyePos = eyePos;
glm::vec3 targetEyePos;
glm::vec3 startLightPos = lightPos;
glm::vec3 targetLightPos;

bool hitPlatform = false;
int platformCount = 0;
vector<glm::mat4> currentCubesModelMatrices;
vector<glm::vec3> currentCubesCoordinates;
vector<glm::mat4> removeModelMatrices;
vector<glm::vec3> removeCoordinates;


int points = 0;
std::string directionText = "Front";


bool gameOver = false;

int midLevelCount = 0;
float midLevely;
vector<int> indexList;

bool isLeftSide = false;
float stopXPosition;

// Holds all state information relevant to a character as loaded using FreeType
struct Character {
    GLuint TextureID;   // ID handle of the glyph texture
    glm::ivec2 Size;    // Size of glyph
    glm::ivec2 Bearing;  // Offset from baseline to left/top of glyph
    GLuint Advance;    // Horizontal offset to advance to next glyph
};

std::map<GLchar, Character> Characters;

// For reading GLSL files
bool ReadDataFromFile(
    const string& fileName, ///< [in]  Name of the shader file
    string&       data)     ///< [out] The contents of the file
{
    fstream myfile;

    // Open the input 
    myfile.open(fileName.c_str(), std::ios::in);

    if (myfile.is_open())
    {
        string curLine;

        while (getline(myfile, curLine))
        {
            data += curLine;
            if (!myfile.eof())
            {
                data += "\n";
            }
        }

        myfile.close();
    }
    else
    {
        return false;
    }

    return true;
}

GLuint createVS(const char* shaderName)
{
    string shaderSource;

    string filename(shaderName);
    if (!ReadDataFromFile(filename, shaderSource))
    {
        cout << "Cannot find file name: " + filename << endl;
        exit(-1);
    }

    GLint length = shaderSource.length();
    const GLchar* shader = (const GLchar*) shaderSource.c_str();

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &shader, &length);
    glCompileShader(vs);

    char output[1024] = {0};
    glGetShaderInfoLog(vs, 1024, &length, output);
    printf("VS compile log: %s\n", output);

	return vs;
}

GLuint createFS(const char* shaderName)
{
    string shaderSource;

    string filename(shaderName);
    if (!ReadDataFromFile(filename, shaderSource))
    {
        cout << "Cannot find file name: " + filename << endl;
        exit(-1);
    }

    GLint length = shaderSource.length();
    const GLchar* shader = (const GLchar*) shaderSource.c_str();

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &shader, &length);
    glCompileShader(fs);

    char output[1024] = {0};
    glGetShaderInfoLog(fs, 1024, &length, output);
    printf("FS compile log: %s\n", output);

	return fs;
}

void initFonts(int windowWidth, int windowHeight)
{
    // Set OpenGL options
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glm::mat4 projection = glm::ortho(0.0f, static_cast<GLfloat>(windowWidth), 0.0f, static_cast<GLfloat>(windowHeight));
    glUseProgram(gProgram[2]);
    glUniformMatrix4fv(glGetUniformLocation(gProgram[2], "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    // FreeType
    FT_Library ft;
    // All functions return a value different than 0 whenever an error occurred
    if (FT_Init_FreeType(&ft))
    {
        std::cout << "ERROR::FREETYPE: Could not init FreeType Library" << std::endl;
    }

    // Load font as face
    FT_Face face;
    if (FT_New_Face(ft, "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 0, &face))
    //if (FT_New_Face(ft, "/usr/share/fonts/truetype/gentium-basic/GenBkBasR.ttf", 0, &face)) // you can use different fonts
    {
        std::cout << "ERROR::FREETYPE: Failed to load font" << std::endl;
    }

    // Set size to load glyphs as
    FT_Set_Pixel_Sizes(face, 0, 48);

    // Disable byte-alignment restriction
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); 

    // Load first 128 characters of ASCII set
    for (GLubyte c = 0; c < 128; c++)
    {
        // Load character glyph 
        if (FT_Load_Char(face, c, FT_LOAD_RENDER))
        {
            std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
            continue;
        }
        // Generate texture
        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RED,
                face->glyph->bitmap.width,
                face->glyph->bitmap.rows,
                0,
                GL_RED,
                GL_UNSIGNED_BYTE,
                face->glyph->bitmap.buffer
                );
        // Set texture options
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        // Now store character for later use
        Character character = {
            texture,
            glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
            glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
            (GLuint) face->glyph->advance.x
        };
        Characters.insert(std::pair<GLchar, Character>(c, character));
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    // Destroy FreeType once we're finished
    FT_Done_Face(face);
    FT_Done_FreeType(ft);

    //
    // Configure VBO for texture quads
    //
    glGenBuffers(1, &gTextVBO);
    glBindBuffer(GL_ARRAY_BUFFER, gTextVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 6 * 4, NULL, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void initShaders()
{
	// Create the programs

    gProgram[0] = glCreateProgram();
	gProgram[1] = glCreateProgram();
	gProgram[2] = glCreateProgram();
    gProgram[3] = glCreateProgram();

	// Create the shaders for both programs

    GLuint vs1 = createVS("vert.glsl"); // for cube shading
    GLuint fs1 = createFS("frag.glsl");

	GLuint vs2 = createVS("vert2.glsl"); // for border shading
	GLuint fs2 = createFS("frag2.glsl");

	GLuint vs3 = createVS("vert_text.glsl");  // for text shading
	GLuint fs3 = createFS("frag_text.glsl");

    GLuint vs4 = createVS("vert.glsl");  // for platform shading (same as cube)
	GLuint fs4 = createFS("frag.glsl");



	// Attach the shaders to the programs

	glAttachShader(gProgram[0], vs1);
	glAttachShader(gProgram[0], fs1);

	glAttachShader(gProgram[1], vs2);
	glAttachShader(gProgram[1], fs2);

	glAttachShader(gProgram[2], vs3);
	glAttachShader(gProgram[2], fs3);

    glAttachShader(gProgram[3], vs4);
    glAttachShader(gProgram[3], fs4);

	// Link the programs

    for (int i = 0; i < 4; ++i)
    {
        glLinkProgram(gProgram[i]);
        GLint status;
        glGetProgramiv(gProgram[i], GL_LINK_STATUS, &status);

        if (status != GL_TRUE)
        {
            cout << "Program link failed: " << i << endl;
            exit(-1);
        }
    }

	// Get the locations of the uniform variables from both programs

	for (int i = 0; i < 2; ++i)
	{
		modelingMatrixLoc[i] = glGetUniformLocation(gProgram[i], "modelingMatrix");
		viewingMatrixLoc[i] = glGetUniformLocation(gProgram[i], "viewingMatrix");
		projectionMatrixLoc[i] = glGetUniformLocation(gProgram[i], "projectionMatrix");
		eyePosLoc[i] = glGetUniformLocation(gProgram[i], "eyePos");
		lightPosLoc[i] = glGetUniformLocation(gProgram[i], "lightPos");
		kdLoc[i] = glGetUniformLocation(gProgram[i], "kd");
        glUseProgram(gProgram[i]);
        glUniformMatrix4fv(modelingMatrixLoc[i], 1, GL_FALSE, glm::value_ptr(modelingMatrix));
        glUniform3fv(eyePosLoc[i], 1, glm::value_ptr(eyePos));
        glUniform3fv(lightPosLoc[i], 1, glm::value_ptr(lightPos));
        glUniform3fv(kdLoc[i], 1, glm::value_ptr(kdCubes));
	}
    modelingMatrixLoc[2] = glGetUniformLocation(gProgram[3], "modelingMatrix");
    viewingMatrixLoc[2] = glGetUniformLocation(gProgram[3], "viewingMatrix");
    projectionMatrixLoc[2] = glGetUniformLocation(gProgram[3], "projectionMatrix");
    eyePosLoc[2] = glGetUniformLocation(gProgram[3], "eyePos");
    lightPosLoc[2] = glGetUniformLocation(gProgram[3], "lightPos");
    kdLoc[2] = glGetUniformLocation(gProgram[3], "kd");
    
    
    gridModel = glm::scale(gridModel,glm::vec3(3.0,1.0/3.0,3.0));
    gridModel = glm::translate(gridModel,glm::vec3(-1.f,-42.2f,-1.f));    
    glUseProgram(gProgram[3]);
    glUniformMatrix4fv(modelingMatrixLoc[2], 1, GL_FALSE, glm::value_ptr(gridModel));
    glUniform3fv(eyePosLoc[2], 1, glm::value_ptr(eyePos));
    glUniform3fv(lightPosLoc[2], 1, glm::value_ptr(lightPos));
    glUniform3fv(kdLoc[2], 1, glm::value_ptr(kdGround));
}

// VBO setup for drawing a cube and its borders
void initVBO()
{
    GLuint vao;
    glGenVertexArrays(1, &vao);
    assert(vao > 0);
    glBindVertexArray(vao);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	assert(glGetError() == GL_NONE);

	glGenBuffers(1, &gVertexAttribBuffer);
	glGenBuffers(1, &gIndexBuffer);

	assert(gVertexAttribBuffer > 0 && gIndexBuffer > 0);

	glBindBuffer(GL_ARRAY_BUFFER, gVertexAttribBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gIndexBuffer);

    GLuint indices[] = {
        0, 1, 2, // front
        3, 0, 2, // front
        4, 7, 6, // back
        5, 4, 6, // back
        0, 3, 4, // left
        3, 7, 4, // left
        2, 1, 5, // right
        6, 2, 5, // right
        3, 2, 7, // top 0,3,3 3,3,3 0,3,0
        2, 6, 7, // top 3,3,3 3,3,0 0,3,0
        0, 4, 1, // bottom
        4, 5, 1,  // bottom
    };

    GLuint indicesLines[] = {
        7, 3, 2, 6, // top 0,3,0 - 0,3,3 - 3,3,3 - 3,3,0
        4, 5, 1, 0, // bottom 0,0,0 - 3,0,0 - 3,0,3 - 0,0,3
        2, 1, 5, 6, // right
        5, 4, 7, 6, // back
        0, 1, 2, 3, // front 0,0,3 - 3,0,3 - 3,3,3 - 0,3,3
        0, 3, 7, 4, // left
        9, 8, 11, 10, // tops 1   1,3,0 - 1,3,3 - 2,3,3 - 2,3,0 
        13, 12, 14, 15, // tops 2   3,3,1 - 0,3,1 - 0,3,2 - 3,3,2 
        16, 17, 19, 18, // bottoms 1   1,0,0 - 1,0,3 - 2,0,3 - 2,0,0
        20, 21, 23, 22, // bottoms 2
        21, 13, 15, 23, // rights 1     3,0,1 - 3,3,1 - 3,3,2 - 3,0,2
        24, 25, 27, 26, // rights 2     3,1,0 - 3,1,3 - 3,2,3 - 3,2,0
        16, 9, 18, 10, // backs 1
        24, 28, 26, 29, // backs 2   3 1 0
        8, 17, 19, 11, // fronts 1  1, 3, 3   1, 0, 3  2, 0, 3  2, 3, 3  
        30, 25, 27, 31, // fronts 2  0, 1, 3   3, 1, 3  0, 2, 3   3, 2, 3
        12, 20, 14, 22, // lefts 1
        28, 30, 29, 31, // lefts 2
        
        
        32,33,34,35,
        36,37,38,39,
        40,41,42,43,
        
        44,45,46,47,
        48,49,50,51,
        52,53,54,55,

        56,57,58,59,
        60,61,62,63,
        64,65,66,67,
        68,69,70,71,
        72,73,74,75,
        76,77,78,79,

        80,81,82,83,
        84,85,86,87,
        88,89,90,91,
        92,93,94,95,
        96,97,98,99,
        100,101,102,103,
        104,105,106,107,
        108,109,110,111,
        112,113,114,115,
        116,117,118,119,
        120,121,122,123,
        124,125,126,127,

        128,129,130,131,
        132,133,134,135,
        136,137,138,139,
        140,141,142,143,
        144,145,146,147,
        148,149,150,151,
        152,153,154,155,
        156,157,158,159,
        160,161,162,163,
        164,165,166,167,
        168,169,170,171,
        172,173,174,175,
        176,177,178,179,
        180,181,182,183,
        184,185,186,187,
        188,189,190,191,
        192,193,194,195,
        196,197,198,199,
    };

    GLfloat vertexPos[] = {
        0, 0, 3, // 0: bottom-left-front 0  
        3, 0, 3, // 3: bottom-right-front 1
        3, 3, 3, // 2: top-right-front 2
        0, 3, 3, // 3: top-left-front 3
        0, 0, 0, // 0: bottom-left-back 4 
        3, 0, 0, // 3: bottom-right-back 5
        3, 3, 0, // 2: top-right-back 6
        0, 3, 0, // 3: top-left-back 7
        1, 3, 3, // top-left-front x:1 8
        1, 3, 0, // top-left-back  x:1 9 
        2, 3, 0, // top-left-front x:2 10
        2, 3, 3, // top-left-back  x:2 11
        0, 3, 1, // top-left-front z:1 12
        3, 3, 1, // top-left-back  z:1 13 
        0, 3, 2, // top-left-front z:2 14
        3, 3, 2, // top-left-back  z:2 15
        1, 0, 0, // bottom-left-front x:1 16
        1, 0, 3, // bottom-left-back  x:1 17 
        2, 0, 0, // bottom-left-front x:2 18
        2, 0, 3, // bottom-left-back  x:2 19
        0, 0, 1, // bottom-left-front z:1 20
        3, 0, 1, // bottom-left-back  z:1 21 
        0, 0, 2, // bottom-left-front z:2 22
        3, 0, 2,  // bottom-left-back  z:2 23
        3, 1, 0, // right-bottom-front y:1 24
        3, 1, 3, // right-bottom-back  y:1 25 
        3, 2, 0, // right-bottom-front y:2 26
        3, 2, 3,  // right-bottom-back  y:2 27
        0, 1, 0, // back-bottom-front y:1 28
        0, 2, 0, // back-bottom-front y:2 29
        0, 1, 3, // front-bottom-front y:1 30
        0, 2, 3, // front-bottom-front y:2 31

        0.333f, 0, 3, //32
        0.333f, 3, 3, 
        0.333f, 3, 0, 
        0.333f, 0, 0, 
        
        0.666f, 0, 3, //32
        0.666f, 3, 3, 
        0.666f, 3, 0, 
        0.666f, 0, 0, 
        
        1.333f, 0, 3, //36
        1.333f, 3, 3, 
        1.333f, 3, 0, 
        1.333f, 0, 0, 
    
        1.666f, 0, 3, //40
        1.666f, 3, 3, 
        1.666f, 3, 0, 
        1.666f, 0, 0, 

        2.333f, 0, 3, //44
        2.333f, 3, 3, 
        2.333f, 3, 0, 
        2.333f, 0, 0, 

        2.666f, 0, 3, //48
        2.666f, 3, 3, 
        2.666f, 3, 0, 
        2.666f, 0, 0, 


        
        0, 3,0.333f, 
        3, 3, 0.333f, 
        3, 0, 0.333f, 
        0, 0, 0.333f, 

        0, 3,0.666f, 
        3, 3, 0.666f, 
        3, 0, 0.666f, 
        0, 0, 0.666f,    
        
        0, 3, 1.333f, //60
        3, 3, 1.333f, 
        3, 0, 1.333f, 
        0, 0, 1.333f, 

        0, 3, 1.666f, 
        3, 3, 1.666f, 
        3, 0, 1.666f, 
        0, 0, 1.666f, 

        0, 3, 2.333f, 
        3, 3, 2.333f, 
        3, 0, 2.333f, 
        0, 0, 2.333f, 
        
        0, 3, 2.666f, 
        3, 3, 2.666f, 
        3, 0, 2.666f, 
        0, 0, 2.666f, 

        0, 3, 2, 
        3, 3, 2, 
        3, 3, 2, 
        0, 3, 2, 

        1, 3, 3, //80
        1, 3, 3, 
        1, 3, 0, 
        1, 3, 0,   
        
        0, 3, 1, 
        3, 3, 1, 
        3, 3, 1, 
        0, 3, 1, 

        0, 3, 3, 
        3, 3, 3, 
        3, 3, 0,  
        0, 3, 0, 

        0, 0, 3,
        3, 0, 3, 
        3, 0, 0,  
        0, 0, 0, 

        2.f, 3, 3, 
        2.f, 3, 3, 
        2.f, 3, 0,  
        2.f, 3, 0, 

        2.33f, 3, 3, //100
        2.33f, 3, 3, 
        2.33f, 0, 3,  
        2.33f, 0, 3, 

        2.0f, 3, 3, 
        2.0f, 3, 3, 
        2.0f, 0, 3,  
        2.0f, 0, 3, 

        1.666f, 3, 3, // 112
        1.666f, 3, 3, 
        1.666f, 0, 3,  
        1.666f, 0, 3, 

        1.f, 3, 3,  //116
        1.f, 3, 3, 
        1.f, 0, 3,  
        1.f, 0, 3, 

        0.f, 3, 3, // 120
        0.f, 3, 3, 
        0.f, 0, 3,  
        0.f, 0, 3, 

        3.f, 3, 3, // 124 
        3.f, 3, 3, 
        3.f, 0, 3,  
        3.f, 0, 3, 


        2.33f, 3, 0, // 128 
        2.33f, 3, 0, 
        2.33f, 0, 0,  
        2.33f, 0, 0, 

        2.0f, 3, 0, // 132 
        2.0f, 3, 0, 
        2.0f, 0, 0,  
        2.0f, 0, 0, 

        1.666f, 3, 0, // 136 
        1.666f, 3, 0, 
        1.666f, 0, 0,  
        1.666f, 0, 0, 

        1.f, 3, 0, // 140 
        1.f, 3, 0, 
        1.f, 0, 0,  
        1.f, 0, 0, 

        0.f, 3, 0, // 144 
        0.f, 3, 0, 
        0.f, 0, 0,  
        0.f, 0, 0, 

        3.f, 3, 0, // 148 
        3.f, 3, 0, 
        3.f, 0, 0,  
        3.f, 0, 0, 


        0, 3, 2.33f, // 152 
        0, 3, 2.33f, 
        0, 0, 2.33f,  
        0, 0, 2.33f, 

        0, 3, 2.0f, // 156 
        0, 3, 2.0f, 
        0, 0, 2.0f,  
        0, 0, 2.0f, 

        0, 3, 1.666f, // 160 
        0, 3, 1.666f, 
        0, 0, 1.666f,  
        0, 0, 1.666f, 

        0, 3, 1.f, // 164 
        0, 3, 1.f, 
        0, 0, 1.f,  
        0, 0, 1.f, 

        0.f, 3, 0, // 168 
        0.f, 3, 0, 
        0.f, 0, 0,  
        0.f, 0, 0, 

        0, 3, 3.f, // 172 
        0, 3, 3.f, 
        0, 0, 3.f,  
        0, 0, 3.f, 

        
        3, 3, 2.33f, // 176 
        3, 3, 2.33f, 
        3, 0, 2.33f,  
        3, 0, 2.33f, 

        3, 3, 2.0f, // 180 
        3, 3, 2.0f, 
        3, 0, 2.0f,  
        3, 0, 2.0f, 

        3, 3, 1.666f, // 184 
        3, 3, 1.666f, 
        3, 0, 1.666f,  
        3, 0, 1.666f, 

        3, 3, 1.f, // 188 
        3, 3, 1.f, 
        3, 0, 1.f,  
        3, 0, 1.f, 

        3, 3, 0, // 192 
        3, 3, 0, 
        3, 0, 0,  
        3, 0, 0, 

        3, 3, 3.f, // 196 
        3, 3, 3.f, 
        3, 0, 3.f,  
        3, 0, 3.f, 


        


    
        };

    GLfloat vertexNor[] = {
         1.0,  1.0,  1.0, // 0: unused
         0.0, -1.0,  0.0, // 1: bottom
         0.0,  0.0,  1.0, // 2: front
         1.0,  1.0,  1.0, // 3: unused
        -1.0,  0.0,  0.0, // 4: left
         1.0,  0.0,  0.0, // 5: right
         0.0,  0.0, -1.0, // 6: back 
         0.0,  1.0,  0.0, // 7: top
    };

	gVertexDataSizeInBytes = sizeof(vertexPos);
	gNormalDataSizeInBytes = sizeof(vertexNor);
    gTriangleIndexDataSizeInBytes = sizeof(indices);
    gLineIndexDataSizeInBytes = sizeof(indicesLines);
    int allIndexSize = gTriangleIndexDataSizeInBytes + gLineIndexDataSizeInBytes;

	glBufferData(GL_ARRAY_BUFFER, gVertexDataSizeInBytes + gNormalDataSizeInBytes, 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, gVertexDataSizeInBytes, vertexPos);
	glBufferSubData(GL_ARRAY_BUFFER, gVertexDataSizeInBytes, gNormalDataSizeInBytes, vertexNor);

	glBufferData(GL_ELEMENT_ARRAY_BUFFER, allIndexSize, 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, gTriangleIndexDataSizeInBytes, indices);
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, gTriangleIndexDataSizeInBytes, gLineIndexDataSizeInBytes, indicesLines);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(gVertexDataSizeInBytes));
}

void init() 
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    // polygon offset is used to prevent z-fighting between the cube and its borders
    glPolygonOffset(0.5, 0.5);
    glEnable(GL_POLYGON_OFFSET_FILL);

    initShaders();
    initVBO();
    initFonts(gWidth, gHeight);
}

void drawCube()
{
    for (glm::mat4 cube : currentCubesModelMatrices) {
        glUseProgram(gProgram[0]);
        glUniformMatrix4fv(modelingMatrixLoc[0], 1, GL_FALSE, glm::value_ptr(cube));
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    }
}

void drawCubeEdges()
{
    for (glm::mat4 cube : currentCubesModelMatrices) {       
        glLineWidth(3);

        glUseProgram(gProgram[1]);
        glUniformMatrix4fv(modelingMatrixLoc[1], 1, GL_FALSE, glm::value_ptr(cube));
        for (int i = 0; i < 6; ++i)
        {
            glDrawElements(GL_LINES, 52, GL_UNSIGNED_INT, BUFFER_OFFSET(gTriangleIndexDataSizeInBytes + i * 4 * sizeof(GLuint)));
        }
    }
}

void drawNewCube() { 
    glUseProgram(gProgram[0]);
    glUniformMatrix4fv(modelingMatrixLoc[0], 1, GL_FALSE, glm::value_ptr(modelingMatrix));
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
}

void drawNewCubeEdges() { 
    glLineWidth(3);
    glUseProgram(gProgram[1]);
    glUniformMatrix4fv(modelingMatrixLoc[1], 1, GL_FALSE, glm::value_ptr(modelingMatrix));
    for (int i = 0; i < 6; ++i)
    {
        glDrawElements(GL_LINES, 52, GL_UNSIGNED_INT, BUFFER_OFFSET(gTriangleIndexDataSizeInBytes + i * 4 * sizeof(GLuint)));
    }
}

void drawPlatform()
{
    glUseProgram(gProgram[3]);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
}

void drawPlatformEdges()
{

    glLineWidth(3);
    glUseProgram(gProgram[1]);
    glUniformMatrix4fv(modelingMatrixLoc[1], 1, GL_FALSE, glm::value_ptr(gridModel));
    for (int i = 18; i < 60; ++i)
    {
	    glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_INT, BUFFER_OFFSET(gTriangleIndexDataSizeInBytes + i * 4 * sizeof(GLuint)));
    }
}

void renderText(const std::string& text, GLfloat x, GLfloat y, GLfloat scale, glm::vec3 color)
{
    // Activate shader for text
    glUseProgram(gProgram[2]);
    glUniform3f(glGetUniformLocation(gProgram[2], "textColor"), color.x, color.y, color.z);
    glActiveTexture(GL_TEXTURE0);

    glm::mat4 projection = glm::ortho(0.0f, static_cast<GLfloat>(gWidth), 0.0f, static_cast<GLfloat>(gHeight));
    glUniformMatrix4fv(glGetUniformLocation(gProgram[2], "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    // Render characters
    for (std::string::const_iterator c = text.begin(); c != text.end(); ++c)
    {
        Character ch = Characters[*c];

        GLfloat xpos = x + ch.Bearing.x * scale;
        GLfloat ypos = y - (ch.Size.y - ch.Bearing.y) * scale;

        GLfloat w = ch.Size.x * scale;
        GLfloat h = ch.Size.y * scale;

        GLfloat vertices[6][4] = {
            { xpos,     ypos + h,   0.0, 0.0 },
            { xpos,     ypos,       0.0, 1.0 },
            { xpos + w, ypos,       1.0, 1.0 },

            { xpos,     ypos + h,   0.0, 0.0 },
            { xpos + w, ypos,       1.0, 1.0 },
            { xpos + w, ypos + h,   1.0, 0.0 }
        };

        glBindTexture(GL_TEXTURE_2D, ch.TextureID);
        glBindBuffer(GL_ARRAY_BUFFER, gTextVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        x += (ch.Advance >> 6) * scale;
    }
    glBindTexture(GL_TEXTURE_2D, 0);
}


void display()
{
    glClearColor(0, 0, 0, 1);
    glClearDepth(1.0f);
    glClearStencil(0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    drawCubeEdges();
    drawCube();
    drawPlatform();
    drawNewCubeEdges();
    drawNewCube();
    drawPlatformEdges();
    renderText(directionText, (gWidth/2)-250, gHeight-100, 0.5, glm::vec3(1, 1, 0));
    renderText("Points: " + std::to_string(points), (gWidth/2)+150, gHeight-100, 0.5, glm::vec3(1, 1, 0));
    if (isA) {
        renderText("A", (gWidth/2)-250, gHeight-200, 0.5, glm::vec3(1, 1, 0));
    }
    else if (isD) {
        renderText("D", (gWidth/2)-250, gHeight-200, 0.5, glm::vec3(1, 1, 0));
    }
    else if (isH) {
        renderText("H", (gWidth/2)-250, gHeight-200, 0.5, glm::vec3(1, 1, 0));
    }
    else if (isK) {
        renderText("K", (gWidth/2)-250, gHeight-200, 0.5, glm::vec3(1, 1, 0));
    }
    else if (isW) {
        renderText("W", (gWidth/2)-250, gHeight-200, 0.5, glm::vec3(1, 1, 0));
    }
    else if (isS) {
        renderText("S", (gWidth/2)-250, gHeight-200, 0.5, glm::vec3(1, 1, 0));
    }
    if (mod == -1) {
        renderText("Speed: Freezed", (gWidth/2)-250, gHeight-150, 0.35, glm::vec3(1, 1, 0));
    }
    else if (mod == 15) {
        renderText("Speed: Fast", (gWidth/2)-250, gHeight-150, 0.35, glm::vec3(1, 1, 0));
    }  
    else if (mod == 45) {
        renderText("Speed: Normal", (gWidth/2)-250, gHeight-150, 0.35, glm::vec3(1, 1, 0));
    } 
    else if (mod == 75) {
        renderText("Speed: Slow", (gWidth/2)-250, gHeight-150, 0.35, glm::vec3(1, 1, 0));
    }
    else if (mod == 105) {
        renderText("Speed: Slower", (gWidth/2)-250, gHeight-150, 0.35, glm::vec3(1, 1, 0));
    }    
    if (gameOver){
        renderText("GAME OVER!", (gWidth/2)-225, gHeight-400, 1.5, glm::vec3(1, 1, 0));
    }


    assert(glGetError() == GL_NO_ERROR);
}

void reshape(GLFWwindow* window, int w, int h)
{
    w = w < 1 ? 1 : w;
    h = h < 1 ? 1 : h;

    gWidth = w;
    gHeight = h;

    glViewport(0, 0, w, h);

	// Use perspective projection

	float fovyRad = (float) (45.0 / 180.0) * M_PI;
	projectionMatrix = glm::perspective(fovyRad, gWidth / (float) gHeight, 1.0f, 100.0f);

    // always look toward (0, 0, 0)
	viewingMatrix = glm::lookAt(eyePos, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));

    for (int i = 0; i < 2; ++i)
    {
        glUseProgram(gProgram[i]);
        glUniformMatrix4fv(projectionMatrixLoc[i], 1, GL_FALSE, glm::value_ptr(projectionMatrix));
        glUniformMatrix4fv(viewingMatrixLoc[i], 1, GL_FALSE, glm::value_ptr(viewingMatrix));
    }
    glUseProgram(gProgram[3]);
    glUniformMatrix4fv(projectionMatrixLoc[2], 1, GL_FALSE, glm::value_ptr(projectionMatrix));
    glUniformMatrix4fv(viewingMatrixLoc[2], 1, GL_FALSE, glm::value_ptr(viewingMatrix));
}



bool gonnaHitCubeSideways(const vector<glm::vec3>& currentCubesCoordinates, float x, float y, float z) {
    for (glm::vec3 currCubes : currentCubesCoordinates){
        float y_epsilon = fabs(currCubes.y - y);
        float x_epsilon = fabs(currCubes.x - x);
        float z_epsilon = fabs(currCubes.z - z);
        if ((y_epsilon < 3.f) && (x_epsilon < 3.f) && (z_epsilon < 3.f)) {
            isLeftSide = (currCubes.x > x); // D is not allowed in this case, otherwise A is not allowed.
            return true;        
        }
    }
    return false;  
}

void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if ((key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE) && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    if ((key == GLFW_KEY_H) && action == GLFW_PRESS){
        isH = true;
        eye_k--;
        if (eye_k <= -4 || eye_k >= 4){
            eye_k = 0;
        }
        if (eye_k == -1 || eye_k == 3){
            targetEyePos = eye_Left;
            targetLightPos = light_Left;
            directionText = "Left";
        }
        else if (eye_k == -2 || eye_k == 2){
            targetEyePos = eye_Back;
            targetLightPos = light_Back;
            directionText = "Back";
        }
        else if (eye_k == -3 || eye_k == 1){
            targetEyePos = eye_Right;
            targetLightPos = light_Right;
            directionText = "Right";
        }
        else{
            targetEyePos = eye_Front;
            targetLightPos = light_Front;
            directionText = "Front";
        }
        isTransitioning = true;
    }
    else if ((key == GLFW_KEY_H) && action == GLFW_RELEASE){
        isH = false;
    }
    else if ((key == GLFW_KEY_K) && action == GLFW_PRESS){
        isK = true;
        eye_k++;
        if (eye_k <= -4 || eye_k >= 4){
            eye_k = 0;
        }
        if (eye_k == -1 || eye_k == 3){
            targetEyePos = eye_Left;
            targetLightPos = light_Left;
            directionText = "Left";
        }
        else if (eye_k == -2 || eye_k == 2){
            targetEyePos = eye_Back;
            targetLightPos = light_Back;
            directionText = "Back";
        }
        else if (eye_k == -3 || eye_k == 1){
            targetEyePos = eye_Right;
            targetLightPos = light_Right;
            directionText = "Right";
        }
        else{
            targetEyePos = eye_Front;
            targetLightPos = light_Front;
            directionText = "Front";
        }
        isTransitioning = true;
    }
    else if ((key == GLFW_KEY_K) && action == GLFW_RELEASE){
        isK = false;
    }
    if (!gameOver) {
        if ((key == GLFW_KEY_A) && action == GLFW_PRESS){
            isA = true;
            if (eye_k == -1 || eye_k == 3){ // left
                if (zLimit >= -2.f && !gonnaHitCubeSideways(currentCubesCoordinates, xLimit, yLimit, zLimit - 1.f)){
                    zLimit -= 1.f;
                    modelingMatrix = glm::translate(modelingMatrix, glm::vec3(0.f, 0.f, -1.f));
                    for (int i = 0; i < 2; ++i){
                        glUseProgram(gProgram[i]);
                        glUniformMatrix4fv(modelingMatrixLoc[i], 1, GL_FALSE, glm::value_ptr(modelingMatrix));
                    }
                }
            }
            else if (eye_k == -2 || eye_k == 2){ // back
                if (xLimit <= 2.f && !gonnaHitCubeSideways(currentCubesCoordinates, xLimit + 1.f, yLimit, zLimit)){
                    xLimit += 1.f;
                    modelingMatrix = glm::translate(modelingMatrix, glm::vec3(1.f, 0.f, 0.f));
                    for (int i = 0; i < 2; ++i){
                        glUseProgram(gProgram[i]);
                        glUniformMatrix4fv(modelingMatrixLoc[i], 1, GL_FALSE, glm::value_ptr(modelingMatrix));
                    }
                }
            }
            else if (eye_k == -3 || eye_k == 1){ // right
                if (zLimit <= 2.f && !gonnaHitCubeSideways(currentCubesCoordinates, xLimit, yLimit, zLimit + 1.f)){
                    zLimit += 1.f;
                    modelingMatrix = glm::translate(modelingMatrix, glm::vec3(0.f, 0.f, 1.f));
                    for (int i = 0; i < 2; ++i){
                        glUseProgram(gProgram[i]);
                        glUniformMatrix4fv(modelingMatrixLoc[i], 1, GL_FALSE, glm::value_ptr(modelingMatrix));
                    }
                }
            } 
            else{ // front 
                if (xLimit >= -2.f && !gonnaHitCubeSideways(currentCubesCoordinates, xLimit - 1.f, yLimit, zLimit)){
                    xLimit -= 1.f;
                    modelingMatrix = glm::translate(modelingMatrix, glm::vec3(-1.f, 0.f, 0.f));
                    for (int i = 0; i < 2; ++i){
                        glUseProgram(gProgram[i]);
                        glUniformMatrix4fv(modelingMatrixLoc[i], 1, GL_FALSE, glm::value_ptr(modelingMatrix));
                    }
                }
            }
        }
        else if ((key == GLFW_KEY_A) && action == GLFW_RELEASE){
            isA = false;
        }
        
        else if ((key == GLFW_KEY_D) && action == GLFW_PRESS){
            isD = true;

            if (eye_k == -1 || eye_k == 3){ // left
                if (zLimit <= 2.f && !gonnaHitCubeSideways(currentCubesCoordinates, xLimit, yLimit, zLimit + 1.f)){
                    zLimit += 1.f;
                    modelingMatrix = glm::translate(modelingMatrix, glm::vec3(0.f, 0.f, 1.f));
                    for (int i = 0; i < 2; ++i){
                        glUseProgram(gProgram[i]);
                        glUniformMatrix4fv(modelingMatrixLoc[i], 1, GL_FALSE, glm::value_ptr(modelingMatrix));
                    }
                }
            }
            else if (eye_k == -2 || eye_k == 2){ // back
                if (xLimit >= -2.f && !gonnaHitCubeSideways(currentCubesCoordinates, xLimit - 1.f, yLimit, zLimit)){
                    xLimit -= 1.f;
                    modelingMatrix = glm::translate(modelingMatrix, glm::vec3(-1.f, 0.f, 0.f));
                    for (int i = 0; i < 2; ++i){
                        glUseProgram(gProgram[i]);
                        glUniformMatrix4fv(modelingMatrixLoc[i], 1, GL_FALSE, glm::value_ptr(modelingMatrix));
                    }
                }
            }
            else if (eye_k == -3 || eye_k == 1){ // right
                if (zLimit >= -2.f && !gonnaHitCubeSideways(currentCubesCoordinates, xLimit, yLimit, zLimit - 1.f)){
                    zLimit -= 1.f;
                    modelingMatrix = glm::translate(modelingMatrix, glm::vec3(0.f, 0.f, -1.f));
                    for (int i = 0; i < 2; ++i){
                        glUseProgram(gProgram[i]);
                        glUniformMatrix4fv(modelingMatrixLoc[i], 1, GL_FALSE, glm::value_ptr(modelingMatrix));
                    }
                }
            } 
            else{ // front 
                if (xLimit <= 2.f && !gonnaHitCubeSideways(currentCubesCoordinates, xLimit + 1.f, yLimit, zLimit)){
                    xLimit += 1.f;
                    modelingMatrix = glm::translate(modelingMatrix, glm::vec3(1.f, 0.f, 0.f));
                    for (int i = 0; i < 2; ++i){
                        glUseProgram(gProgram[i]);
                        glUniformMatrix4fv(modelingMatrixLoc[i], 1, GL_FALSE, glm::value_ptr(modelingMatrix));
                    }
                }
            }
        }
        else if ((key == GLFW_KEY_D) && action == GLFW_RELEASE){
            isD = false;
        }
        else if ((key == GLFW_KEY_S) && action == GLFW_PRESS){
            isS = true;
            if (mod == -1) {
                mod = 105;
            }
            else if (mod <= 15){mod = 15;}
            else{mod -= 30;}
        }
        else if ((key == GLFW_KEY_S) && action == GLFW_RELEASE){
            isS = false;
        }
        else if ((key == GLFW_KEY_W) && action == GLFW_PRESS){
            isW = true;
            if (mod >= 105 || mod == -1) {
                mod = -1;
            }
            else{mod += 30;}
        }
        else if ((key == GLFW_KEY_W) && action == GLFW_RELEASE){
            isW = false;
        }
    }
}

bool timer(int &value, int &mod){
    if (mod == -1){return false;} // stop cube
    return (abs(value--) % mod == 0);
}

bool hitCube(vector<glm::vec3> &currentCubesCoordinates, float x, float y, float z) {
    for (glm::vec3 currCubes : currentCubesCoordinates){
        float y_epsilon = fabs(currCubes.y - y);
        float x_epsilon = fabs(currCubes.x - x);
        float z_epsilon = fabs(currCubes.z - z);
        if ((y_epsilon <= 3.f) && (x_epsilon < 3.f) && (z_epsilon < 3.f)) {
            return true;
        }
    }
    return false;
}

void mainLoop(GLFWwindow* window)
{
    while (!glfwWindowShouldClose(window))
    {
        if (isTransitioning){
            if(transitionSpeed < 1.f){
                transitionVector = glm::mix(eyePos, targetEyePos, transitionSpeed);
                lightPos = glm::mix(lightPos, targetLightPos, transitionSpeed);
                transitionSpeed += 0.08f;
            }
            else{
                transitionVector = targetEyePos;
                eyePos = targetEyePos;
                lightPos = targetLightPos;
                isTransitioning = false;
                transitionSpeed = 0.f;
            }
            viewingMatrix = glm::lookAt(transitionVector, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
            for (int i = 0; i < 2; ++i) {
                glUseProgram(gProgram[i]);
                glUniform3fv(lightPosLoc[i], 1, glm::value_ptr(lightPos));
                glUniformMatrix4fv(viewingMatrixLoc[i], 1, GL_FALSE, glm::value_ptr(viewingMatrix));
            }
            glUseProgram(gProgram[3]);
            glUniformMatrix4fv(viewingMatrixLoc[2], 1, GL_FALSE, glm::value_ptr(viewingMatrix));
        }
        // moving down 
        if (yLimit >= ground_coord && !hitCube(currentCubesCoordinates, xLimit, yLimit, zLimit)){
            if (timer(k, mod)){ // timer logic 
                yLimit -= falling_speed;
                if (yLimit < ground_coord){
                    modelingMatrix = glm::translate(modelingMatrix, glm::vec3(0.f, yLimit - ground_coord, 0.f));
                    hitPlatform = true;
                }
                else{
                    modelingMatrix = glm::translate(modelingMatrix, glm::vec3(0.f, -falling_speed, 0.f));
                }
                for (int i = 0; i < 2; ++i){
                    glUseProgram(gProgram[i]);
                    glUniformMatrix4fv(modelingMatrixLoc[i], 1, GL_FALSE, glm::value_ptr(modelingMatrix));
                }  
            }
        }
        if (hitPlatform) { // cube hits platform 
            currentCubesModelMatrices.push_back(modelingMatrix);
            modelingMatrix = baseModelMatrix;
            currentCubesCoordinates.push_back(glm::vec3(xLimit, yLimit, zLimit));
            xLimit = 0.f;
            yLimit = 0.f;
            zLimit = 0.f;
            platformCount++;
            hitPlatform = false;
        }
        else if (hitCube(currentCubesCoordinates, xLimit, yLimit, zLimit)){ // cube hits cube
            if (yLimit >= 0.f){ // GAME OVER
                gameOver = true;
            }
            currentCubesModelMatrices.push_back(modelingMatrix);
            modelingMatrix = baseModelMatrix;
            currentCubesCoordinates.push_back(glm::vec3(xLimit, yLimit, zLimit));
            xLimit = 0.f;
            yLimit = 0.f;
            zLimit = 0.f;
        }
        else if (gonnaHitCubeSideways(currentCubesCoordinates, xLimit, yLimit, zLimit)){
            if ((isA && (isLeftSide)) || (isD && (!isLeftSide))){
                currentCubesModelMatrices.push_back(modelingMatrix);
                modelingMatrix = baseModelMatrix;
                currentCubesCoordinates.push_back(glm::vec3(xLimit, yLimit, zLimit));
                xLimit = 0.f;
                yLimit = 0.f;
                zLimit = 0.f;
            }
        }
        if (platformCount == 9) { // BOOM 
            for (int i = 0; i < currentCubesModelMatrices.size(); ++i){
                glm::mat4 currModel = currentCubesModelMatrices[i];
                if (currentCubesCoordinates[i].y == -13){
                    removeModelMatrices.push_back(currModel);
                    removeCoordinates.push_back(currentCubesCoordinates[i]);
                }
            }
            for (glm::mat4 removeModel : removeModelMatrices){
                currentCubesModelMatrices.erase(std::remove(currentCubesModelMatrices.begin(), currentCubesModelMatrices.end(), removeModel), currentCubesModelMatrices.end());
            }
            for (glm::vec3 removeCoord : removeCoordinates){
                currentCubesCoordinates.erase(std::remove(currentCubesCoordinates.begin(), currentCubesCoordinates.end(), removeCoord), currentCubesCoordinates.end());
            }
            removeModelMatrices.clear();
            removeCoordinates.clear();
            platformCount = 0;
            for (glm::mat4 &remainingModel : currentCubesModelMatrices){
                remainingModel = glm::translate(remainingModel, glm::vec3(0.f, -3.f, 0.f));
            }
            
            for (glm::vec3 &remainingCoord : currentCubesCoordinates){
                remainingCoord.y -= 3.f;
                if (remainingCoord.y == -13.f){
                    platformCount++;
                }
            }
            if(!gameOver){
                points += 243;
            }
        }
        
        // mid level BOOM
        int index = 0;
        for (glm::vec3 coord : currentCubesCoordinates){
            for (int i = index+1; i < currentCubesCoordinates.size(); ++i){
                if (coord.y == currentCubesCoordinates[i].y){
                    midLevelCount++; 
                }
            }
            if (midLevelCount == 8){
                midLevely = coord.y;
                break;
            }
            else{
                midLevelCount = 0;
            }
            index++;
        }
        if (midLevelCount == 8) {
            for (int i = 0; i < currentCubesModelMatrices.size(); ++i){
                glm::mat4 currModel = currentCubesModelMatrices[i];
                if (currentCubesCoordinates[i].y == midLevely){
                    removeModelMatrices.push_back(currModel);
                    removeCoordinates.push_back(currentCubesCoordinates[i]);
                }
            }
            for (glm::mat4 removeModel : removeModelMatrices){
                currentCubesModelMatrices.erase(std::remove(currentCubesModelMatrices.begin(), currentCubesModelMatrices.end(), removeModel), currentCubesModelMatrices.end());
            }
            for (glm::vec3 removeCoord : removeCoordinates){
                currentCubesCoordinates.erase(std::remove(currentCubesCoordinates.begin(), currentCubesCoordinates.end(), removeCoord), currentCubesCoordinates.end());
            }
            removeModelMatrices.clear();
            removeCoordinates.clear();
            midLevelCount = 0;
            index = 0;
            for (glm::vec3 &remainingCoord : currentCubesCoordinates){
                if (remainingCoord.y > midLevely){
                    remainingCoord.y -= 3.f;
                    indexList.push_back(index);
                }
                index++;
            }
            for (int index : indexList){
                currentCubesModelMatrices[index] = glm::translate(currentCubesModelMatrices[index], glm::vec3(0.f, -3.f, 0.f));
            }
            if(!gameOver){
                points += 243;
            }
        }
        
        display();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

int main(int argc, char** argv) // Create Main Function For Bringing It All Together
{
    GLFWwindow* window;
    if (!glfwInit())
    {
        exit(-1);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(gWidth, gHeight, "tetrisGL", NULL, NULL);

    if (!window)
    {
        glfwTerminate();
        exit(-1);
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Initialize GLEW to setup the OpenGL Function pointers
    if (GLEW_OK != glewInit())
    {
        std::cout << "Failed to initialize GLEW" << std::endl;
        return EXIT_FAILURE;
    }

    char rendererInfo[512] = {0};
    strcpy(rendererInfo, (const char*) glGetString(GL_RENDERER));
    strcat(rendererInfo, " - ");
    strcat(rendererInfo, (const char*) glGetString(GL_VERSION));
    glfwSetWindowTitle(window, rendererInfo);

    init();

    glfwSetKeyCallback(window, keyboard);
    glfwSetWindowSizeCallback(window, reshape);

    reshape(window, gWidth, gHeight); // need to call this once ourselves
    mainLoop(window); // this does not return unless the window is closed

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}



