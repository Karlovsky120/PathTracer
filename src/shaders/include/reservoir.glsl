#include "random.glsl"
#include "structs.glsl"

void updateReservoirAt(inout Reservoir res,
					   int i,
					   float weight,
					   vec3 position,
					   vec4 normal,
					   float emissionLum,
					   int lightIdx,
					   float pHat,
					   float w,
					   inout Random random)
{
	res.samples[i].sumWeights += weight;
	float replacePossibility = weight / res.samples[i].sumWeights;
	if (randFloat(random) < replacePossibility)
	{
		res.samples[i].position_emissionLum = vec4(position, emissionLum);
		res.samples[i].normal = normal;
		res.samples[i].lightIndex = lightIdx;
		res.samples[i].pHat = pHat;
		res.samples[i].w = w;
	}
}

void addSampleToReservoir(inout Reservoir res,
						  vec3 position,
						  vec4 normal,
						  float emissionLum,
						  int lightIdx,
						  float pHat,
						  float sampleP,
						  inout Random random)
{
	float weight = pHat / sampleP;
	res.numStreamSamples += 1;

	for (int i = 0; i < RESERVOIR_SIZE; ++i)
	{
		float w = (res.samples[i].sumWeights + weight) / (res.numStreamSamples * pHat);
		updateReservoirAt(
			res, i, weight, position, normal, emissionLum, lightIdx, pHat, w,
			random
		);
	}
}

void combineReservoirs(inout Reservoir self, Reservoir other, float pHat[RESERVOIR_SIZE], inout Random random)
{
	self.numStreamSamples += other.numStreamSamples;

	for (int i = 0; i < RESERVOIR_SIZE; ++i)
	{
		float weight = pHat[i] * other.samples[i].w * other.numStreamSamples;
		if (weight > 0.0f)
		{
			updateReservoirAt(
				self, i, weight,
				other.samples[i].position_emissionLum.xyz, other.samples[i].normal, other.samples[i].position_emissionLum.w,
				other.samples[i].lightIndex, pHat[i],
				other.samples[i].w, random
			);
		}

		if (self.samples[i].w > 0.0f)
		{
			self.samples[i].w = self.samples[i].sumWeights / (self.numStreamSamples * self.samples[i].pHat);
		}
	}
}

Reservoir newReservoir()
{
	Reservoir result;
	for (int i = 0; i < RESERVOIR_SIZE; ++i)
	{
		result.samples[i].sumWeights = 0.0f;
	}

	result.numStreamSamples = 0;
	return result;
}
