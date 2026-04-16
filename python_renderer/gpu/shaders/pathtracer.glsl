#version 430 core
// ============================================================
// pathtracer.glsl  –  Main compute shader for GPU path tracing
// ============================================================

layout(local_size_x = 16, local_size_y = 16) in;

// ---- output image -------------------------------------------------------
layout(rgba32f, binding = 0) uniform image2D u_accumBuffer;
layout(rgba32f, binding = 1) uniform image2D u_outputImage;

// ---- uniforms ------------------------------------------------------------
uniform int   u_sampleIndex;
uniform int   u_maxBounces;
uniform float u_fireflyClamp;
uniform float u_cameraNear;
uniform float u_cameraFar;
uniform float u_cameraFov;      // vertical FoV in radians
uniform float u_aspectRatio;
uniform mat4  u_cameraMatrix;   // camera-to-world
uniform int   u_frameWidth;
uniform int   u_frameHeight;
uniform int   u_numTriangles;
uniform int   u_hasEnv;
uniform sampler2D u_envMap;

// ---- BVH / geometry buffers ---------------------------------------------
layout(std430, binding = 2) readonly buffer TriangleBuffer {
    // per triangle: 9 vertices + 9 normals + 6 uvs = 24 floats, padded to 28
    float triData[];
};

// ---- material buffer (std430, 64 floats, see material.py) ---------------
layout(std430, binding = 3) readonly buffer MaterialBuffer {
    float matData[];
};

// ---- include shared utilities -------------------------------------------
#include "common.glsl"
#include "brdf.glsl"

// =========================================================================
// Triangle intersection (Möller–Trumbore)
// =========================================================================
struct HitInfo {
    float t;
    vec3  normal;
    vec2  uv;
    int   matID;
    bool  hit;
};

HitInfo intersectTriangle(Ray ray, int triIdx) {
    HitInfo hi;
    hi.hit = false;
    int base = triIdx * 28;

    vec3 v0 = vec3(triData[base+0],  triData[base+1],  triData[base+2]);
    vec3 v1 = vec3(triData[base+3],  triData[base+4],  triData[base+5]);
    vec3 v2 = vec3(triData[base+6],  triData[base+7],  triData[base+8]);
    vec3 n0 = vec3(triData[base+9],  triData[base+10], triData[base+11]);
    vec3 n1 = vec3(triData[base+12], triData[base+13], triData[base+14]);
    vec3 n2 = vec3(triData[base+15], triData[base+16], triData[base+17]);
    vec2 uv0 = vec2(triData[base+18], triData[base+19]);
    vec2 uv1 = vec2(triData[base+20], triData[base+21]);
    vec2 uv2 = vec2(triData[base+22], triData[base+23]);
    // base+24 = matID (packed as float)

    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    vec3 h = cross(ray.direction, edge2);
    float a = dot(edge1, h);
    if (abs(a) < 1e-8) return hi;

    float f = 1.0 / a;
    vec3  s = ray.origin - v0;
    float u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) return hi;

    vec3  q = cross(s, edge1);
    float v = f * dot(ray.direction, q);
    if (v < 0.0 || u + v > 1.0) return hi;

    float t = f * dot(edge2, q);
    if (t < ray.tMin || t > ray.tMax) return hi;

    float w = 1.0 - u - v;
    hi.hit    = true;
    hi.t      = t;
    hi.normal = normalize(w * n0 + u * n1 + v * n2);
    hi.uv     = w * uv0 + u * uv1 + v * uv2;
    hi.matID  = int(triData[base + 24]);
    return hi;
}

HitInfo intersectScene(Ray ray) {
    HitInfo closest;
    closest.hit = false;
    closest.t   = ray.tMax;

    for (int i = 0; i < u_numTriangles; ++i) {
        HitInfo hi = intersectTriangle(ray, i);
        if (hi.hit && hi.t < closest.t) {
            closest = hi;
        }
    }
    return closest;
}

// =========================================================================
// Environment sampling
// =========================================================================
vec3 sampleEnvironment(vec3 dir) {
    if (u_hasEnv == 0) {
        // Simple procedural sky gradient
        float t = 0.5 * (dir.y + 1.0);
        return mix(vec3(1.0, 0.85, 0.7), vec3(0.5, 0.7, 1.0), t);
    }
    vec2 uv = dirToEquirect(dir);
    return texture(u_envMap, uv).rgb;
}

// =========================================================================
// Path-tracing integrator
// =========================================================================
vec3 tracePath(Ray primaryRay, inout uint rngState) {
    vec3 radiance    = vec3(0.0);
    vec3 throughput  = vec3(1.0);
    Ray  ray         = primaryRay;

    for (int bounce = 0; bounce <= u_maxBounces; ++bounce) {
        HitInfo hi = intersectScene(ray);

        if (!hi.hit) {
            radiance += throughput * sampleEnvironment(ray.direction);
            break;
        }

        // ---- fetch material -----------------------------------------------
        int mbase = hi.matID * 64;
        vec3  baseColor        = vec3(matData[mbase+0],  matData[mbase+1],  matData[mbase+2]);
        float baseWeight       = matData[mbase+3];
        vec3  specColor        = vec3(matData[mbase+4],  matData[mbase+5],  matData[mbase+6]);
        float specWeight       = matData[mbase+7];
        float specRoughness    = max(matData[mbase+8], 0.001);
        float specAnisotropy   = matData[mbase+9];
        float ior              = matData[mbase+10];
        float metalness        = matData[mbase+11];
        float emissionWeight   = matData[mbase+41];
        float emissionLum      = matData[mbase+42];
        vec3  emissionColor    = vec3(matData[mbase+44], matData[mbase+45], matData[mbase+46]);
        float coatWeight       = matData[mbase+31];
        float coatRoughness    = max(matData[mbase+32], 0.001);
        float coatIor          = matData[mbase+33];

        vec3 hitPos = ray.origin + hi.t * ray.direction;
        vec3 N      = hi.normal;
        // Flip normal for back-face hits (thin-walled)
        if (dot(N, -ray.direction) < 0.0) N = -N;

        // ---- emission -------------------------------------------------------
        if (emissionWeight > 0.0) {
            radiance += throughput * emissionColor * emissionLum * emissionWeight;
        }

        // ---- coat layer (clear coat BRDF) -----------------------------------
        vec3 coatContrib = vec3(0.0);
        float coatMask   = 0.0;
        if (coatWeight > 0.0) {
            float F0coat = fresnelSchlickScalar(ior2F0(coatIor));
            float cosTheta = max(dot(N, -ray.direction), 0.0);
            coatMask = coatWeight * fresnelSchlick(F0coat, cosTheta);
        }

        // ---- base/specular BRDF evaluation ----------------------------------
        vec3  V   = -ray.direction;
        float F0  = ior2F0(ior);
        vec3  F0c = mix(vec3(F0), baseColor * specColor, metalness);

        // Sample BRDF direction
        float pDiffuse = (1.0 - metalness) * 0.5;
        vec3  sampleDir;
        float pdf;

        float xi = rand(rngState);
        if (xi < pDiffuse) {
            // Cosine-weighted diffuse sample
            sampleDir = cosineHemisphereSample(N, rngState);
            pdf = max(dot(N, sampleDir), 0.0) / PI;
            if (pdf < 1e-6) break;

            vec3 H     = normalize(V + sampleDir);
            vec3 F     = fresnelSchlickRGB(F0c, max(dot(H, V), 0.0));
            vec3 kd    = (1.0 - F) * (1.0 - metalness);
            vec3 diff  = kd * baseColor * baseWeight / PI;
            throughput *= diff * max(dot(N, sampleDir), 0.0) / (pdf * pDiffuse);
        } else {
            // GGX specular sample
            float alpha = specRoughness * specRoughness;
            vec3  H     = sampleGGXVNDF(V, N, alpha, rngState);
            sampleDir   = reflect(-V, H);
            if (dot(N, sampleDir) <= 0.0) break;

            float NdotL = max(dot(N, sampleDir), 0.0);
            float NdotV = max(dot(N, V), 0.0);
            float NdotH = max(dot(N, H), 0.0);
            float VdotH = max(dot(V, H), 0.0);

            float D  = ggxNDF(NdotH, alpha);
            float G  = smithG2(NdotL, NdotV, alpha);
            vec3  F  = fresnelSchlickRGB(F0c, VdotH);
            vec3  spec = specColor * specWeight * D * G * F
                         / max(4.0 * NdotV, 1e-6);

            pdf = D * NdotH / max(4.0 * VdotH, 1e-6);
            if (pdf < 1e-6) break;
            throughput *= spec * NdotL / (pdf * (1.0 - pDiffuse));
        }

        // Apply coat attenuation
        throughput *= (1.0 - coatMask);

        // Firefly clamping
        float lum = dot(throughput, vec3(0.2126, 0.7152, 0.0722));
        if (lum > u_fireflyClamp) {
            throughput *= u_fireflyClamp / lum;
        }

        // ---- advance ray ----------------------------------------------------
        ray.origin    = hitPos + N * 1e-4;
        ray.direction = sampleDir;
        ray.tMin      = 1e-4;
        ray.tMax      = 1e12;

        if (dot(throughput, throughput) < 1e-8) break;
    }
    return radiance;
}

// =========================================================================
// Main entry point
// =========================================================================
void main() {
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);
    if (pixelCoord.x >= u_frameWidth || pixelCoord.y >= u_frameHeight) return;

    uint rngState = pcgHash(
        uint(pixelCoord.x) + uint(u_frameWidth)  * uint(pixelCoord.y)
        + uint(u_sampleIndex) * uint(u_frameWidth * u_frameHeight)
    );

    // ---- generate jittered camera ray ------------------------------------
    float jitterX = rand(rngState) - 0.5;
    float jitterY = rand(rngState) - 0.5;
    vec2 uv = (vec2(pixelCoord) + vec2(0.5) + vec2(jitterX, jitterY))
              / vec2(float(u_frameWidth), float(u_frameHeight));
    uv = uv * 2.0 - 1.0;

    float tanHalfFov = tan(u_cameraFov * 0.5);
    vec3 rayDirCam = normalize(vec3(
        uv.x * u_aspectRatio * tanHalfFov,
        uv.y * tanHalfFov,
        -1.0
    ));
    vec3 rayDir    = normalize((u_cameraMatrix * vec4(rayDirCam, 0.0)).xyz);
    vec3 rayOrigin = (u_cameraMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;

    Ray primaryRay;
    primaryRay.origin    = rayOrigin;
    primaryRay.direction = rayDir;
    primaryRay.tMin      = u_cameraNear;
    primaryRay.tMax      = u_cameraFar;

    vec3 sample = tracePath(primaryRay, rngState);

    // ---- progressive accumulation ----------------------------------------
    vec4 prev = imageLoad(u_accumBuffer, pixelCoord);
    float w   = 1.0 / float(u_sampleIndex + 1);
    vec4 accum = prev * (1.0 - w) + vec4(sample, 1.0) * w;
    imageStore(u_accumBuffer, pixelCoord, accum);

    // ---- ACES tone map & write output ------------------------------------
    vec3 tonemapped = acesTonemap(accum.rgb);
    vec3 gammaCorrected = pow(max(tonemapped, 0.0), vec3(1.0 / 2.2));
    imageStore(u_outputImage, pixelCoord, vec4(gammaCorrected, 1.0));
}
