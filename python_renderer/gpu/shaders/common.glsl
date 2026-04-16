// ============================================================
// common.glsl  –  Shared utility functions
// ============================================================

#ifndef COMMON_GLSL
#define COMMON_GLSL

#define PI      3.14159265358979323846
#define TWO_PI  6.28318530717958647692
#define INV_PI  0.31830988618379067154
#define EPS     1e-6

// ---- Ray struct ---------------------------------------------------------
struct Ray {
    vec3  origin;
    vec3  direction;
    float tMin;
    float tMax;
};

// =========================================================================
// Random number generation (PCG hash)
// =========================================================================
uint pcgHash(uint state) {
    uint x = state * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (x >> 22u) ^ x;
}

float rand(inout uint state) {
    state = pcgHash(state);
    return float(state) * (1.0 / 4294967296.0);
}

vec2 rand2(inout uint state) {
    return vec2(rand(state), rand(state));
}

vec3 rand3(inout uint state) {
    return vec3(rand(state), rand(state), rand(state));
}

// =========================================================================
// Orthonormal basis construction (Frisvad / Duff et al.)
// =========================================================================
mat3 buildTBN(vec3 N) {
    vec3 up = abs(N.y) < 0.9999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T  = normalize(cross(up, N));
    vec3 B  = cross(N, T);
    return mat3(T, B, N);
}

// =========================================================================
// Hemispherical sampling
// =========================================================================
vec3 cosineHemisphereSample(vec3 N, inout uint state) {
    vec2 r   = rand2(state);
    float phi = TWO_PI * r.x;
    float sinTheta = sqrt(r.y);
    float cosTheta = sqrt(1.0 - r.y);
    vec3 local = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    return normalize(buildTBN(N) * local);
}

vec3 uniformHemisphereSample(vec3 N, inout uint state) {
    vec2  r   = rand2(state);
    float phi = TWO_PI * r.x;
    float cosTheta = r.y;
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    vec3  local = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    return normalize(buildTBN(N) * local);
}

// =========================================================================
// GGX / VNDF sampling
// =========================================================================
vec3 sampleGGXVNDF(vec3 V, vec3 N, float alpha, inout uint state) {
    // Transform V to local space
    mat3 tbn   = buildTBN(N);
    vec3 Vlocal = transpose(tbn) * V;

    // Sample the GGX VNDF
    vec2 r = rand2(state);
    vec3 Vh = normalize(vec3(alpha * Vlocal.x, alpha * Vlocal.y, Vlocal.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3  T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) / sqrt(lensq) : vec3(1.0, 0.0, 0.0);
    vec3  T2 = cross(Vh, T1);

    float r1   = r.x;
    float r2   = r.y;
    float phi  = TWO_PI * r1;
    float s    = 0.5 * (1.0 + Vh.z);
    float sinT = sqrt(max(0.0, r2 * (1.0 - s) / (1.0 - s * Vh.z) + s));
    float cosT = sqrt(max(0.0, 1.0 - sinT * sinT));

    vec3 Nh = cosT * Vh + sinT * (cos(phi) * T1 + sin(phi) * T2);
    vec3 Ne = normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));

    // Transform back to world space
    return normalize(tbn * Ne);
}

// =========================================================================
// Fresnel helpers
// =========================================================================
float ior2F0(float ior) {
    float f = (ior - 1.0) / (ior + 1.0);
    return f * f;
}

float fresnelSchlickScalar(float F0) {
    // Called with cosTheta = 1 to get material-independent value
    return F0;
}

float fresnelSchlick(float F0, float cosTheta) {
    float x = 1.0 - cosTheta;
    return F0 + (1.0 - F0) * x * x * x * x * x;
}

vec3 fresnelSchlickRGB(vec3 F0, float cosTheta) {
    float x = 1.0 - cosTheta;
    return F0 + (1.0 - F0) * vec3(x * x * x * x * x);
}

// =========================================================================
// Tone mapping
// =========================================================================
vec3 acesTonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 reinhardTonemap(vec3 x) {
    return x / (1.0 + x);
}

vec3 filmicTonemap(vec3 x) {
    x = max(vec3(0.0), x - 0.004);
    return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06);
}

// =========================================================================
// Color space conversion
// =========================================================================
vec3 linearToSRGB(vec3 c) {
    bvec3 cutoff = lessThan(c, vec3(0.0031308));
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(hi, lo, vec3(cutoff));
}

vec3 sRGBToLinear(vec3 c) {
    bvec3 cutoff = lessThan(c, vec3(0.04045));
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(hi, lo, vec3(cutoff));
}

float luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// =========================================================================
// Equirectangular projection
// =========================================================================
vec2 dirToEquirect(vec3 dir) {
    float u = atan(dir.z, dir.x) / TWO_PI + 0.5;
    float v = asin(clamp(dir.y, -1.0, 1.0)) / PI + 0.5;
    return vec2(u, v);
}

vec3 equirectToDir(vec2 uv) {
    float phi   = (uv.x - 0.5) * TWO_PI;
    float theta = (uv.y - 0.5) * PI;
    return vec3(cos(theta) * cos(phi), sin(theta), cos(theta) * sin(phi));
}

#endif // COMMON_GLSL
