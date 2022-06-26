bool testVisibility(vec3 p1, vec3 p2)
{
	float tMin = 0.001f;
	vec3 dir = p2 - p1;

	isShadowed = true;

	float curTMax = length(dir);
	dir /= curTMax;

	traceRayEXT(
		acc,
		gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
		0xFF,
		0,
		0,
		0,
		p1,
		tMin,
		dir,
		curTMax - 2.0f * tMin,
		0
	);

	return isShadowed;
}
