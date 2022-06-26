#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

struct Random
{
	uint64_t state;
	uint64_t inc;
};

uint randUint(inout Random random)
{
	uint64_t oldState = random.state;
	random.state = oldState * 6364136223846793005ul + random.inc;
	uint xorShifted = uint(((oldState >> 18u) ^ oldState) >> 27u);
	uint rot = uint(oldState >> 59u);
	return (xorShifted >> rot) | (xorShifted << ((-rot) & 31u));
}

Random seedRand(uint64_t seed, uint64_t seq)
{
	Random random;
	random.state = 0;
	random.inc = (seq << 1u) | 1u;
	randUint(random);
	random.state += seed;
	randUint(random);
	return random;
}

float randFloat(inout Random random)
{
	return randUint(random) / 4294967296.0f;
}
