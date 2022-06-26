#extension GL_EXT_ray_tracing : require

#include "include/structs.glsl"
#include "include/reservoir.glsl"
#include "include/brdf.glsl"

layout (binding = 0, set = 0) buffer PointLights
{
	int count;
	PointLight lights[];
} pointLights;

layout (binding = 1, set = 0) buffer TriangleLights
{
	int count;
	TriangleLight lights[];
} triangleLights;

layout (binding = 2, set = 0) buffer AliasTable
{
	int count;
	int padding[3];
	Bucket buckets[];
} aliasTable;

layout (binding = 3, set = 0) uniform Uniforms
{
	RestirUniforms uniforms;
};

layout (binding = 4, set = 0) uniform accelerationStructureEXT acc;

layout (binding = 0, set = 1) uniform sampler2D WorldPositionTexture;
layout (binding = 1, set = 1) uniform sampler2D AlbedoTexture;
layout (binding = 2, set = 1) uniform sampler2D NormalTexture;
layout (binding = 3, set = 1) uniform sampler2D MaterialPropertiesTexture;

layout (binding = 4, set = 1) uniform sampler2D PreviousFrameWorldPositionTexture;
layout (binding = 5, set = 1) uniform sampler2D PreviousFrameAlbedoTexture;
layout (binding = 6, set = 1) uniform sampler2D PreviousFrameNormalTexture;
layout (binding = 7, set = 1) uniform sampler2D PreviousDepthTexture;

layout (binding = 8, set = 1) buffer Reservoirs
{
	Reservoir reservoirs[];
};

layout (binding = 9, set = 1) buffer PreviousFrameReservoirs
{
	Reservoir prevFrameReservoirs[];
};

layout (location = 0) rayPayloadEXT bool isShadowed;
#include "include/visibility.glsl"

vec3 pickPointOnTriangle(float r1, float r2, vec3 p1, vec3 p2, vec3 p3)
{
	float sqrt_r1 = sqrt(r1);
	return (1.0 - sqrt_r1) * p1 + (sqrt_r1 * (1.0 - r2)) * p2 + (r2 * sqrt_r1) * p3;
}

void aliasTableSample(float r1, float r2, out int index, out float probability)
{
	int selected_bucket = min(int(aliasTable.count * r1), aliasTable.count - 1);
	Bucket bucket = aliasTable.buckets[selected_bucket];
	if (bucket.probability > r2)
	{
		index = selected_bucket;
		probability = bucket.originalProbability;
	}
	else
	{
		index = bucket.alias;
		probability = bucket.aliasOriginalProbability;
	}
}

void main()
{
	uvec2 pixel = gl_LaunchIDEXT.xy;
	if (any(greaterThanEqual(pixel, uniforms.screenSize)))
	{
		return;
	}

	vec3 albedo = texelFetch(AlbedoTexture, ivec2(pixel), 0).xyz;
	vec3 normal = texelFetch(NormalTexture, ivec2(pixel), 0).xyz;
	vec2 roughnessMetallic = texelFetch(MaterialPropertiesTexture, ivec2(pixel), 0).xy;
	vec3 worldPos = texelFetch(WorldPositionTexture, ivec2(pixel), 0).xyz;

	float albedoLum =  0.2126f * albedo.r + 0.7152f * albedo.g + 0.0722f * albedo.b;

	Reservoir res = newReservoir();
	Random random = seedRand(uniforms.frame, pixel.y * 10007 + pixel.x);
	if (dot(normal, normal) != 0.0f)
	{
		for (int i = 0; i < uniforms.lightSampleCount; ++i)
		{
			int selected_idx;
			float lightSampleProb;
			aliasTableSample(randFloat(random), randFloat(random), selected_idx, lightSampleProb);

			vec3 lightSamplePos;
			vec4 lightNormal;
			float lightSampleLum;
			int lightSampleIndex;
			if (pointLights.count != 0)
			{
				PointLight light = pointLights.lights[selected_idx];
				lightSamplePos = light.pos.xyz;
				lightSampleLum = light.color_luminance.w;
				lightSampleIndex = selected_idx;
				lightNormal = vec4(0.0f);
			}
			else
			{
				TriangleLight light = triangleLights.lights[selected_idx];
				lightSamplePos = pickPointOnTriangle(randFloat(random), randFloat(random), light.p1.xyz, light.p2.xyz, light.p3.xyz);
				lightSampleLum = light.emission_luminance.w;
				lightSampleIndex = -1 - selected_idx;

				vec3 wi = normalize(worldPos - lightSamplePos);
				vec3 normal = light.normalArea.xyz;
				lightSampleProb /= abs(dot(wi, normal)) * light.normalArea.w;
				lightNormal = vec4(normal, 1.0f);
			}

			float pHat = evaluatePHat(
				worldPos, lightSamplePos, uniforms.cameraPos.xyz,
				normal, lightNormal.xyz, lightNormal.w > 0.5f,
				albedoLum, lightSampleLum, roughnessMetallic.x, roughnessMetallic.y
			);

			addSampleToReservoir(res, lightSamplePos, lightNormal, lightSampleLum, lightSampleIndex, pHat, lightSampleProb, random);
		}
	}

	uint reservoirIndex = pixel.y * uniforms.screenSize.x + pixel.x;

	if ((uniforms.flags & RESTIR_VISIBILITY_REUSE_FLAG) != 0)
	{
		for (int i = 0; i < RESERVOIR_SIZE; i++)
		{
			bool shadowed = testVisibility(worldPos, res.samples[i].position_emissionLum.xyz);

			if (shadowed)
			{
				res.samples[i].w = 0.0f;
				res.samples[i].sumWeights = 0.0f;
			}
		}
	}

	if ((uniforms.flags & RESTIR_TEMPORAL_REUSE_FLAG) != 0)
	{
		vec4 prevFramePos = uniforms.prevFrameProjectionViewMatrix * vec4(worldPos, 1.0f);
		prevFramePos.xyz /= prevFramePos.w;
		prevFramePos.xy = (prevFramePos.xy + 1.0f) * 0.5f * vec2(uniforms.screenSize);
		if (all(greaterThan(prevFramePos.xy, vec2(0.0f))) &&
			all(lessThan(prevFramePos.xy, vec2(uniforms.screenSize))))
		{
			ivec2 prevFrag = ivec2(prevFramePos.xy);

			vec3 positionDiff = worldPos - texelFetch(PreviousFrameWorldPositionTexture, prevFrag, 0).xyz;
			if (dot(positionDiff, positionDiff) < 0.01f)
			{
				vec3 albedoDiff = albedo - texelFetch(PreviousFrameAlbedoTexture, prevFrag, 0).rgb;
				if (dot(albedoDiff, albedoDiff) < 0.01f)
				{
					float normalDot = dot(normal, texelFetch(PreviousFrameNormalTexture, prevFrag, 0).xyz);
					if (normalDot > 0.5f)
					{
						Reservoir prevRes = prevFrameReservoirs[prevFrag.y * uniforms.screenSize.x + prevFrag.x];

						prevRes.numStreamSamples = min(
							prevRes.numStreamSamples, uniforms.temporalSampleCountMultiplier * res.numStreamSamples
						);

						vec2 metallicRoughness = texelFetch(MaterialPropertiesTexture, ivec2(pixel), 0).xy;

						float pHat[RESERVOIR_SIZE];
						for (int i = 0; i < RESERVOIR_SIZE; ++i)
						{
							pHat[i] = evaluatePHat(
								worldPos, prevRes.samples[i].position_emissionLum.xyz, uniforms.cameraPos.xyz,
								normal, prevRes.samples[i].normal.xyz, prevRes.samples[i].normal.w > 0.5f,
								albedoLum, prevRes.samples[i].position_emissionLum.w, metallicRoughness.x, metallicRoughness.y
							);
						}

						combineReservoirs(res, prevRes, pHat, random);
					}
				}
			}
		}
	}

	reservoirs[reservoirIndex] = res;
}

