#ifndef GLSL_STRUCTUS
#define GLSL_STRUCTUS

#define RESERVOIR_SIZE 1

#define RESTIR_VISIBILITY_REUSE_FLAG (1 << 0)
#define RESTIR_TEMPORAL_REUSE_FLAG (1 << 1)

#define METALLIC_ROUGHNESS 0
#define SPECULAR_GLOSSINESS 1

struct LightSample
{
	vec4 position_emissionLum;
	vec4 normal;
	int lightIndex;
	float pHat;
	float sumWeights;
	float w;
};

struct Reservoir
{
	LightSample samples[RESERVOIR_SIZE];
	uint numStreamSamples;
};

struct RestirUniforms
{
	mat4 prevFrameProjectionViewMatrix;
	vec4 cameraPos;
	uvec2 screenSize;
	uint frame;

	uint lightSampleCount;

	uint temporalSampleCountMultiplier;

	float spatialPosThreshold;
	float spatialNormalThreshold;
	uint spatialNeighbors;
	float spatialRadius;

	int flags;
};

struct ModelMatrices
{
	mat4 Transform;
	mat4 TransformInverseTransposed;
};

struct MaterialUniforms
{
	vec4 colorParam;
	vec4 materialParam;

	vec4 emissiveFactor;
	int shadingModel;
	int alphaMode;
	float alphaCutoff;
	float normalTextureScale;
};

struct PointLight
{
	vec4 pos;
	vec4 color_luminance;
};

struct TriangleLight
{
	vec4 p1;
	vec4 p2;
	vec4 p3;
	vec4 emission_luminance;
	vec4 normalArea;
};

struct Bucket
{
	float probability;
	int alias;
	float originalProbability;
	float aliasOriginalProbability;
};

struct LightingPassUniforms
{
	mat4 prevFrameProjectionViewMatrix;
	vec4 cameraPos;
	uvec2 bufferSize;
	float gamma;
};

#endif
