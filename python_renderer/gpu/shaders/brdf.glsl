// ============================================================
// brdf.glsl  –  BRDF evaluation functions for OpenPBR v1.2
// ============================================================

#ifndef BRDF_GLSL
#define BRDF_GLSL

// =========================================================================
// GGX Normal Distribution Function
// =========================================================================
float ggxNDF(float NdotH, float alpha) {
    float a2    = alpha * alpha;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Anisotropic GGX NDF
float ggxNDFAniso(float NdotH, float TdotH, float BdotH,
                  float alphaX, float alphaY)
{
    float ax2 = alphaX * alphaX;
    float ay2 = alphaY * alphaY;
    float d   = (TdotH * TdotH) / ax2 + (BdotH * BdotH) / ay2 + NdotH * NdotH;
    return 1.0 / (PI * alphaX * alphaY * d * d);
}

// =========================================================================
// Smith G2 (height-correlated masking-shadowing)
// =========================================================================
float smithLambda(float NdotX, float alpha) {
    float a2   = alpha * alpha;
    float cos2 = NdotX * NdotX;
    float tan2 = max((1.0 - cos2) / cos2, 0.0);
    return 0.5 * (-1.0 + sqrt(1.0 + a2 * tan2));
}

float smithG1(float NdotX, float alpha) {
    return 1.0 / (1.0 + smithLambda(NdotX, alpha));
}

float smithG2(float NdotL, float NdotV, float alpha) {
    return 1.0 / (1.0 + smithLambda(NdotL, alpha) + smithLambda(NdotV, alpha));
}

// =========================================================================
// Cook-Torrance specular BRDF
// =========================================================================
vec3 cookTorranceSpecular(
    vec3  F0,
    float NdotL,
    float NdotV,
    float NdotH,
    float VdotH,
    float roughness)
{
    float alpha = roughness * roughness;
    float D  = ggxNDF(NdotH, alpha);
    float G  = smithG2(NdotL, NdotV, alpha);
    vec3  F  = fresnelSchlickRGB(F0, VdotH);
    float denom = max(4.0 * NdotL * NdotV, 1e-6);
    return D * G * F / denom;
}

// =========================================================================
// Lambert diffuse BRDF
// =========================================================================
vec3 lambertDiffuse(vec3 albedo) {
    return albedo * INV_PI;
}

// =========================================================================
// OpenPBR layer blending
// =========================================================================

// Blend diffuse and specular using energy conservation
vec3 openPBRBlend(
    vec3  baseColor,
    float baseWeight,
    float metalness,
    vec3  specColor,
    float specWeight,
    float specRoughness,
    float ior,
    float coatWeight,
    float coatRoughness,
    float coatIor,
    vec3  V,
    vec3  N,
    vec3  L)
{
    vec3  H     = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    if (NdotL <= 0.0) return vec3(0.0);

    // Base F0
    float f0scalar = ior2F0(ior);
    vec3  F0       = mix(vec3(f0scalar), baseColor * specColor, metalness);

    // Coat layer
    float coatF0     = ior2F0(coatIor);
    float coatFresnel = coatWeight * fresnelSchlick(coatF0, VdotH);
    float coatAlpha  = coatRoughness * coatRoughness;
    float coatD  = ggxNDF(NdotH, coatAlpha);
    float coatG  = smithG2(NdotL, NdotV, coatAlpha);
    vec3  coatSpec = vec3(coatWeight * coatD * coatG * coatFresnel
                         / max(4.0 * NdotL * NdotV, 1e-6));

    // Base specular
    vec3  spec  = specWeight * cookTorranceSpecular(F0, NdotL, NdotV, NdotH, VdotH, specRoughness);

    // Diffuse with metalness mask
    vec3  F      = fresnelSchlickRGB(F0, VdotH);
    vec3  kd     = (1.0 - F) * (1.0 - metalness);
    vec3  diff   = baseWeight * kd * lambertDiffuse(baseColor);

    return NdotL * ((diff + spec) * (1.0 - coatFresnel) + coatSpec);
}

// =========================================================================
// Thin-film iridescence approximation
// =========================================================================
vec3 thinFilmFresnel(float cosTheta, float thickness, float ior) {
    // Simplified thin-film interference (first-order)
    float lambda_r = 700.0;
    float lambda_g = 546.0;
    float lambda_b = 436.0;
    float phase = 2.0 * PI * ior * thickness * cosTheta;
    vec3  F0 = vec3(ior2F0(ior));
    vec3  iridescence = vec3(
        0.5 + 0.5 * cos(phase / lambda_r),
        0.5 + 0.5 * cos(phase / lambda_g),
        0.5 + 0.5 * cos(phase / lambda_b)
    );
    return mix(fresnelSchlickRGB(F0, cosTheta), iridescence, 0.5);
}

// =========================================================================
// Fuzz / sheen BRDF (fabric)
// =========================================================================
float charlieSheen(float NdotH, float roughness) {
    float invR = 1.0 / roughness;
    float cos2h = NdotH * NdotH;
    float sin2h = max(1.0 - cos2h, 0.0078125);
    return (2.0 + invR) * pow(sin2h, invR * 0.5) / (2.0 * PI);
}

vec3 sheenBRDF(vec3 sheenColor, float sheenRoughness,
               float NdotL, float NdotV, float NdotH)
{
    float D = charlieSheen(NdotH, sheenRoughness);
    // Simple visibility term for sheen
    float G = 1.0 / (4.0 * (NdotL + NdotV - NdotL * NdotV));
    return sheenColor * D * G * NdotL;
}

#endif // BRDF_GLSL
