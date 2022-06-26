#define M_PI 3.1415926535897932384626433832795

float schlickFresnel(float cos)
{
	float m = clamp(1 - cos, 0.0, 1.0);
	float mSquared = m * m;
	return mSquared * mSquared * m;
}

float GTR2(float NdotH, float a)
{
	float aSquared = a * a;
	float t = 1.0 + (aSquared - 1.0) * NdotH * NdotH;
	return aSquared / (M_PI * t * t);
}

float smithG_GGX(float NdotV, float alphaG)
{
	float a = alphaG * alphaG;
	float b = NdotV * NdotV;
	return 1.0 / (abs(NdotV) + max(sqrt(a + b - a * b), 0.0001));
}

float disneyBrdfDiffuseFactor(float cosIn, float cosOut, float cosInHalf, float roughness, float metallic)
{
	float fresnelIn = schlickFresnel(cosIn);
	float fresnelOut = schlickFresnel(cosOut);
	float fresnelDiffuse90 = 0.5 + 2.0 * cosInHalf * cosInHalf * roughness;
	float fresnelDiffuse = mix(1.0, fresnelDiffuse90, fresnelIn) * mix(1.0, fresnelDiffuse90, fresnelOut);
	return fresnelDiffuse * (1.0f - metallic) / M_PI;
}

vec2 disneyBrdfSpecularFactors(float cosIn,
							   float cosOut,
							   float cosHalf,
							   float cosInHalf,
							   float roughness,
							   float metallic)
{
	float fresnelInHalf = schlickFresnel(cosInHalf);

	float a = max(0.001, pow(roughness, 2.0));
	float Ds = GTR2(cosHalf, a);

	float Gs;
	Gs = smithG_GGX(cosIn, a);
	Gs *= smithG_GGX(cosOut, a);

	return vec2(fresnelInHalf, Gs * Ds);
}

vec3 disneyBrdfSpecular(float cosIn,
					    float cosOut,
					    float cosHalf,
					    float cosInHalf,
					    vec3 albedo,
					    float roughness,
						float metallic)
{
	vec2 factors = disneyBrdfSpecularFactors(cosIn, cosOut, cosHalf, cosInHalf, roughness, metallic);

	vec3 specularColor = mix(vec3(0.04f), albedo, metallic);
	vec3 Fs = mix(specularColor, vec3(1.0), factors.x);

	return Fs * factors.y;
}

float disneyBrdfSpecularLuminance(float cosIn,
								  float cosOut,
								  float cosHalf,
								  float cosInHalf,
								  float luminance,
								  float roughness,
								  float metallic)
{
	vec2 factors = disneyBrdfSpecularFactors(cosIn, cosOut, cosHalf, cosInHalf, roughness, metallic);

	float specularLuminance = mix(0.04f, luminance, metallic);
	float Fs = mix(specularLuminance, 1.0f, factors.x);

	return Fs * factors.y;
}

vec3 disneyBrdfColor(float cosIn,
					 float cosOut,
					 float cosHalf,
					 float cosInHalf,
					 vec3 albedo,
					 float roughness,
					 float metallic)
{
	if (cosIn < 0.0f)
	{
		return vec3(0.0f);
	}

	vec3 diffuse = albedo * disneyBrdfDiffuseFactor(cosIn, cosOut, cosInHalf, roughness, metallic);
	vec3 specular = disneyBrdfSpecular(cosIn, cosOut, cosHalf, cosInHalf, albedo, roughness, metallic);

	return diffuse + specular;
}

float disneyBrdfLuminance(float cosIn,
						  float cosOut,
						  float cosHalf,
						  float cosInHalf,
						  float albedoLuminance,
						  float roughness,
						  float metallic)
{
	if (cosIn < 0.0f)
	{
		return 0.0f;
	}

	float diffuse = albedoLuminance * disneyBrdfDiffuseFactor(cosIn, cosOut, cosInHalf, roughness, metallic);
	float specular = disneyBrdfSpecularLuminance(cosIn, cosOut, cosHalf, cosInHalf, albedoLuminance, roughness, metallic);

	return diffuse + specular;
}

float evaluatePHat(vec3 worldPos,
				   vec3 lightPos,
				   vec3 camPos,
				   vec3 normal,
				   vec3 lightNormal,
				   bool useLightNormal,
				   float albedoLum,
				   float emissionLum,
				   float roughness,
				   float metallic)
{
	vec3 wi = lightPos - worldPos;
	if (dot(wi, normal) < 0.0f)
	{
		return 0.0f;
	}

	float sqrDist = dot(wi, wi);
	wi /= sqrt(sqrDist);
	vec3 wo = normalize(vec3(camPos) - worldPos);

	float cosIn = dot(normal, wi);
	float cosOut = dot(normal, wo);
	vec3 halfVec = normalize(wi + wo);
	float cosHalf = dot(normal, halfVec);
	float cosInHalf = dot(wi, halfVec);

	float geometry = cosIn / sqrDist;
	if (useLightNormal)
	{
		geometry *= abs(dot(wi, lightNormal));
	}

	return emissionLum * disneyBrdfLuminance(cosIn, cosOut, cosHalf, cosInHalf, albedoLum, roughness, metallic) * geometry;
}

vec3 evaluatePHatFull(vec3 worldPos,
					  vec3 lightPos,
					  vec3 camPos,
					  vec3 normal,
					  vec3 lightNormal,
					  bool useLightNormal,
					  vec3 albedo,
					  vec3 emission,
					  float roughness,
					  float metallic)
{
	vec3 wi = lightPos - worldPos;
	if (dot(wi, normal) < 0.0f)
	{
		return vec3(0.0f);
	}

	float sqrDist = dot(wi, wi);
	wi /= sqrt(sqrDist);
	vec3 wo = normalize(vec3(camPos) - worldPos);

	float cosIn = dot(normal, wi);
	float cosOut = dot(normal, wo);
	vec3 halfVec = normalize(wi + wo);
	float cosHalf = dot(normal, halfVec);
	float cosInHalf = dot(wi, halfVec);

	float geometry = cosIn / sqrDist;
	if (useLightNormal)
	{
		geometry *= abs(dot(wi, lightNormal));
	}

	return emission * disneyBrdfColor(cosIn, cosOut, cosHalf, cosInHalf, albedo, roughness, metallic) * geometry;
}
