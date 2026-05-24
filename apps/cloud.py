"""
Real-time Volumetric Cloud Raymarching — Pure Python / ModernGL / Offscreen FBO


Implements, all in a single fragment shader running on a fullscreen quad:
  - Volumetric raymarching with constant step size
  - 3D value-noise FBM density field with high-frequency erosion
  - Density threshold + remap for crisp cloud silhouette
  - Beer's law transmittance
  - Nested light-march toward the sun for self-shadowing
  - Henyey-Greenstein anisotropic phase function
  - Procedural blue-noise dithering (interleaved gradient noise) with temporal jitter
  - Soft mouse-orbit camera
  - Time-of-day control (sunrise → noon → sunset) driving sun position and palette
  - Optional MediaPipe hand-tracking: horizontal hand position drives time of day

Same plumbing pattern as voxel_car.py: standalone ModernGL context with an
offscreen FBO, read back as QImage each frame and blitted onto a plain QWidget
inside a QGraphicsProxyWidget so it lives happily on a QGraphicsScene.

Dependencies: PySide6, moderngl, numpy
Optional:     opencv-python, mediapipe   (for hand control)
"""

import math
import time
import random
import threading
import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout, QGraphicsItem
from PySide6.QtCore import Qt, QTimer, QPointF, QRect
from PySide6.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush, QImage, QMouseEvent
)

import moderngl


# ─────────────────────────────────────────────
#  Shaders
# ─────────────────────────────────────────────
VERT_SHADER = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

# Fragment shader. Same pipeline as before, but with several changes that
# give the cloud a defined silhouette like the reference photograph:
#   1. Higher MAX_STEPS / smaller MARCH_SIZE — better sampling along the ray
#   2. Density THRESHOLD + remap — turns the FBM field into "cloud or not cloud"
#      instead of a smooth fog
#   3. Sharper envelope falloff so the cloud has a clear silhouette
#   4. Higher absorption — more dramatic self-shadowing contrast
#   5. Sky uses a 3-stop gradient (horizon → mid → zenith) so sunset palettes
#      with strong purples/pinks read properly
FRAG_SHADER = """
#version 330

uniform vec2  u_resolution;
uniform float u_time;
uniform int   u_frame;
uniform vec3  u_cam_pos;
uniform vec3  u_cam_target;
uniform vec3  u_sun_dir;
uniform vec3  u_sun_color;
uniform vec3  u_sky_top;
uniform vec3  u_sky_mid;
uniform vec3  u_sky_horizon;
uniform vec3  u_cloud_shadow;       // tint where the cloud is self-shadowed
uniform vec3  u_cloud_light;        // tint where the cloud is lit
uniform float u_absorption;
uniform float u_aniso;
uniform float u_cloud_radius;
uniform float u_density_scale;
uniform float u_density_threshold;  // 0..1 — bigger = wispier silhouette
uniform float u_edge_sharpness;     // controls envelope falloff hardness
uniform float u_warp_strength;      // domain-warp strength — turbulent swirls
uniform float u_detail_sharpness;   // post-threshold power — detached wisps
uniform float u_storm;              // 0..1 — storm intensity (darkens + thickens)
uniform vec3  u_envelope_stretch;   // per-axis stretch (x,y,z) — wider during storms
uniform vec3  u_storm_sky;          // ultra-dark sky color at full storm
uniform vec3  u_storm_cloud_dark;   // shadow tint for storm clouds
uniform vec3  u_storm_cloud_lit;    // moon-lit edge tint for storm clouds
uniform vec3  u_moon_dir;           // direction toward the moon (replaces sun at night)
uniform float u_moon_strength;      // 0..1 — how much moonlight contributes
uniform float u_lightning;          // 0..1 — current lightning flash brightness
uniform vec3  u_lightning_color;    // tint of the lightning flash
uniform float u_lightning_radius;   // glow radius — small for localized strikes

// Polyline joints defining the bolt's jagged path. Each strike fills the
// first u_bolt_count entries; consecutive joints with the same branch_id
// form a connected segment, joints with a *different* branch_id start a
// new branch (so we don't draw a line between unrelated joints).
#define MAX_BOLT_JOINTS 96
uniform int   u_bolt_count;                       // active joint count
uniform vec3  u_bolt_joints[MAX_BOLT_JOINTS];     // .xyz = position
uniform int   u_bolt_branch_id[MAX_BOLT_JOINTS];  // branch grouping

out vec4 frag;

#define PI 3.14159265359
#define MAX_STEPS         140
#define MAX_STEPS_LIGHT   8
#define MARCH_SIZE        0.12
#define LIGHT_MARCH_SIZE  0.24

// ── Hash + 3D value noise ─────────────────────────────────
float hash13(vec3 p3) {
    p3 = fract(p3 * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float noise3(vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash13(p + vec3(0,0,0));
    float n100 = hash13(p + vec3(1,0,0));
    float n010 = hash13(p + vec3(0,1,0));
    float n110 = hash13(p + vec3(1,1,0));
    float n001 = hash13(p + vec3(0,0,1));
    float n101 = hash13(p + vec3(1,0,1));
    float n011 = hash13(p + vec3(0,1,1));
    float n111 = hash13(p + vec3(1,1,1));
    float nx00 = mix(n000, n100, f.x);
    float nx10 = mix(n010, n110, f.x);
    float nx01 = mix(n001, n101, f.x);
    float nx11 = mix(n011, n111, f.x);
    float nxy0 = mix(nx00, nx10, f.y);
    float nxy1 = mix(nx01, nx11, f.y);
    return mix(nxy0, nxy1, f.z) * 2.0 - 1.0;
}

// Standard FBM — soft, hill-like noise. Used both as a body component
// (mixed with billow) and as the domain-warp displacement field.
float fbm(vec3 p, int octaves) {
    vec3 q = p;
    float f = 0.0;
    float scale = 0.5;
    float factor = 2.02;
    for (int i = 0; i < 8; i++) {
        if (i >= octaves) break;
        f += scale * noise3(q);
        q *= factor;
        factor += 0.21;
        scale *= 0.5;
    }
    return f;
}

// Billow FBM — same skeleton, but |noise|. Each octave bottoms out at zero
// instead of crossing through it, producing rounded billow peaks like real
// cumulus. Slower amplitude decay (0.58 vs 0.5) preserves high-frequency
// detail through the summation.
float fbm_billow(vec3 p, int octaves) {
    vec3 q = p;
    float f = 0.0;
    float scale = 0.55;
    float factor = 2.17;
    float norm = 0.0;
    for (int i = 0; i < 8; i++) {
        if (i >= octaves) break;
        f += scale * (abs(noise3(q)) * 2.0 - 1.0);  // [-1, +1] but billow-shaped
        norm += scale;
        q = q * factor + vec3(13.7, 41.3, 7.1);     // offset breaks lattice axes
        factor += 0.13;
        scale *= 0.58;                              // slower decay = sharper
    }
    return f / max(norm, 1e-3);
}

// Density field. Goal: keep the single cohesive cloud from the previous
// version, but give it crisper internal turbulence and crinkly edges.
//
// Three additions vs. plain FBM:
//   1. mix(standard, billow): rounded peaks for cumulus texture without
//      shifting the field's mean (so the threshold stays calibrated).
//   2. Domain warp: distort the lookup point by a smooth low-octave noise.
//      The displacement is multiplied by the envelope so it tapers to zero
//      at the cloud boundary — this is what keeps the silhouette intact
//      while still creating swirls inside.
//   3. Edge sharpening: pow() applied near the silhouette boundary only,
//      so the interior keeps its body but the edges develop wispy fingers
//      and small detached puffs.
float scene(vec3 p, bool lightSample) {
    // Wind drift — same direction as before, applied uniformly so the whole
    // cloud translates rather than shearing apart.
    vec3 q = p + u_time * 0.18 * vec3(1.0, -0.1, -0.8);

    // Envelope (compute first — used to taper the warp).
    // The stretch vector lets us turn the bounding sphere into an ellipsoid
    // — during storms we elongate horizontally (x,z) much more than
    // vertically (y) so the cloud spreads across the sky rather than just
    // ballooning into a fatter sphere. We divide *position* by the stretch
    // (not the noise), so the texture inside the cloud stays at the same
    // frequency — only the silhouette grows.
    vec3 pe = p / max(u_envelope_stretch, vec3(0.001));
    float dist = length(pe);
    float envelope = 1.0 - smoothstep(u_cloud_radius * 0.6,
                                      u_cloud_radius * (1.0 + 0.6 / u_edge_sharpness),
                                      dist);

    // Domain warp — 2-octave smooth offset. Multiplied by `envelope` so the
    // displacement vanishes outside the cloud (silhouette stays intact).
    // No time term here — the body noise `q` already drifts with wind, and
    // adding a second time term to the warp domain would compound and let
    // the pattern wander out of the bounded region.
    int warpOct = lightSample ? 2 : 3;
    vec3 wp = q * 0.7;
    vec3 warp = vec3(
        fbm(wp + vec3( 0.0,  0.0,  0.0), warpOct),
        fbm(wp + vec3(31.4, 11.2,  7.7), warpOct),
        fbm(wp + vec3(83.1, 47.9, 23.5), warpOct)
    );
    q += warp * u_warp_strength * envelope;

    // Body noise: blend smooth FBM with billow. The standard term keeps
    // the mean density where the threshold expects it; the billow term
    // adds the rounded crisp peaks. 0.6 weight on billow seems to be the
    // sweet spot — more makes the field too dark, less loses the texture.
    int octaves = lightSample ? 4 : 7;
    float fSmooth = fbm(q, octaves);
    float fBillow = fbm_billow(q, octaves);
    float f = mix(fSmooth, fBillow, 0.6);

    // Assemble density.
    float density = f * 0.6 + 0.5;
    density *= envelope;

    // Threshold + remap (unchanged from before — the silhouette knob).
    density = max(0.0, density - u_density_threshold);
    density /= max(1e-3, 1.0 - u_density_threshold);

    // Edge-only sharpening. We power the density toward zero in regions
    // where `envelope` is in its falloff band, so interior cores keep
    // their volume but edge wisps get pinched into fingers. `edgeMask`
    // is 1 in the falloff band, 0 deep inside or fully outside — gives
    // us the detached-puff look without hollowing out the cloud's body.
    if (!lightSample && u_detail_sharpness > 1.0) {
        float edgeMask = 4.0 * envelope * (1.0 - envelope);  // peaks at 0.5
        float p_exp = mix(1.0, u_detail_sharpness, edgeMask);
        density = pow(density, p_exp);
    }

    return density * u_density_scale;
}

float beersLaw(float dist, float absorption) {
    return exp(-dist * absorption);
}

float henyeyGreenstein(float g, float mu) {
    float gg = g * g;
    return (1.0 / (4.0 * PI)) *
           ((1.0 - gg) / pow(1.0 + gg - 2.0 * g * mu, 1.5));
}

// ── Bolt polyline helpers ─────────────────────────────────
//
// The bolt is stored as N joints with a branch_id per joint. Two
// consecutive joints with the same branch_id form a line segment of the
// channel. A change in branch_id means the next segment is the start of
// a *different* branch — we must NOT connect across that boundary, or
// we'd draw long straight lines across the cloud where branches fork.
//
// boltDistance returns the closest distance from world point `p` to any
// segment of the polyline, plus the direction along that segment (out)
// for back-light direction calculations.

float boltDistance(vec3 p, out vec3 outClosest, out vec3 outDir) {
    float minDist = 1e6;
    outClosest = vec3(0.0);
    outDir = vec3(0.0, -1.0, 0.0);
    int count = min(u_bolt_count, MAX_BOLT_JOINTS);
    for (int i = 0; i < MAX_BOLT_JOINTS - 1; i++) {
        if (i + 1 >= count) break;
        // Skip the segment if the two joints belong to different branches.
        if (u_bolt_branch_id[i] != u_bolt_branch_id[i + 1]) continue;
        vec3 a = u_bolt_joints[i];
        vec3 b = u_bolt_joints[i + 1];
        vec3 ab = b - a;
        float abLen2 = max(1e-6, dot(ab, ab));
        float t = clamp(dot(p - a, ab) / abLen2, 0.0, 1.0);
        vec3 closest = a + ab * t;
        float d = length(p - closest);
        if (d < minDist) {
            minDist = d;
            outClosest = closest;
            outDir = ab / sqrt(abLen2);
        }
    }
    return minDist;
}

// Same, but for a 3D point's distance to the bolt as seen from a view
// ray. Returns minimum ray-to-segment distance (continuous version of
// the previous discrete-sample loop). Also returns the t along the ray
// where the closest point is, so callers can reject hits behind the
// camera.
float boltDistanceToRay(vec3 ro, vec3 rd, out float outRayT) {
    float minDist = 1e6;
    outRayT = 0.0;
    int count = min(u_bolt_count, MAX_BOLT_JOINTS);
    for (int i = 0; i < MAX_BOLT_JOINTS - 1; i++) {
        if (i + 1 >= count) break;
        if (u_bolt_branch_id[i] != u_bolt_branch_id[i + 1]) continue;
        vec3 a = u_bolt_joints[i];
        vec3 b = u_bolt_joints[i + 1];
        // Closest distance between two lines (one infinite — the ray —
        // and one finite, the bolt segment). Derivation: minimize
        //   |ro + s·d1 - (a + t·d2)|²
        // with d1 = rd (unit), d2 = b - a. Setting partials to zero
        // gives the system
        //   [ 1   -B ] [s]   [ -C ]
        //   [ -B   E ] [t] = [ -F ]
        // where B = d1·d2, C = d1·r, E = d2·d2, F = d2·r, r = ro - a.
        // The closed-form solution is:
        //   denom = E - B²
        //   t = (F - B·C) / denom            (clamp to [0,1])
        //   s = B·t - C                      (clamp to [0, ∞))
        vec3 d2 = b - a;
        vec3 r  = ro - a;
        float E = dot(d2, d2);
        float F = dot(d2, r);
        float C = dot(rd, r);
        float B = dot(rd, d2);
        float denom = E - B * B;
        float tt, s;
        if (denom > 1e-6) {
            tt = clamp((F - B * C) / denom, 0.0, 1.0);
            s = B * tt - C;
        } else {
            // Parallel — pick segment start, just compute ray param.
            tt = 0.0;
            s = -C;
        }
        // If the optimal s is behind the camera, snap to s=0 and find
        // the closest segment point to ro.
        if (s < 0.0) {
            s = 0.0;
            tt = clamp(F / max(E, 1e-6), 0.0, 1.0);
        }
        vec3 pRay = ro + rd * s;
        vec3 pSeg = a + d2 * tt;
        float d = length(pRay - pSeg);
        if (d < minDist) {
            minDist = d;
            outRayT = s;
        }
    }
    return minDist;
}

float lightmarch(vec3 position) {
    vec3 lightDir = normalize(u_sun_dir);
    float totalDensity = 0.0;
    for (int s = 0; s < MAX_STEPS_LIGHT; s++) {
        // Slight step growth: cover more depth into the cloud cheaply.
        float step = LIGHT_MARCH_SIZE * (1.0 + float(s) * 0.15);
        vec3 lp = position + lightDir * step * float(s);
        float d = scene(lp, true);
        totalDensity += max(0.0, d) * step;
    }
    return beersLaw(totalDensity, u_absorption);
}

float ign(vec2 fragCoord) {
    return fract(52.9829189 * fract(0.06711056 * fragCoord.x
                                  + 0.00583715 * fragCoord.y));
}

vec4 raymarch(vec3 ro, vec3 rd, float offset) {
    float depth = MARCH_SIZE * offset;
    vec3 p = ro + depth * rd;

    float totalTransmittance = 1.0;
    float lightEnergy = 0.0;
    float boltEnergy = 0.0;  // lightning glow accumulated from inside the cloud

    // Phase term is constant along the ray — hoist it.
    float mu = dot(rd, normalize(u_sun_dir));
    float phase = henyeyGreenstein(u_aniso, mu);

    for (int i = 0; i < MAX_STEPS; i++) {
        float density = scene(p, false);
        if (density > 0.0) {
            float lightT = lightmarch(p);
            float luminance = density * phase;
            totalTransmittance *= lightT;
            lightEnergy += totalTransmittance * luminance;

            // Lightning emission: now uses the full polyline. The closest
            // segment determines distance + back-light direction. Two
            // physical-ish contributions:
            //
            //   (a) Inverse-cube-ish falloff with distance to the bolt
            //       channel — handles the "near the bolt is bright"
            //       behaviour. This is roughly r⁻² but softened at the
            //       origin to avoid divide-by-near-zero singularities.
            //
            //   (b) A directional back-light term that boosts samples on
            //       the *side* of the cloud the bolt faces. We do a tiny
            //       3-tap density march from the sample toward the bolt's
            //       closest point: if there's a clear shot (low density
            //       between sample and bolt), this is a "front" sample —
            //       boost it. If there's a lot of cloud in the way, the
            //       bolt's light is occluded — attenuate.
            //
            // Together these make the cloud light up *from the inside*
            // with proper falloff *and* with side-of-cloud preference,
            // exactly matching how internal lightning illumination looks
            // in real photographs.
            if (u_lightning > 0.001) {
                vec3 closest, segDir;
                float boltDist = boltDistance(p, closest, segDir);

                // (a) sharp distance falloff
                float r = max(0.05, u_lightning_radius);
                float k = boltDist / r;
                float boltFalloff = 1.0 / (1.0 + k * k * k * 8.0);

                // (b) short transmittance march toward the bolt — 3 taps
                // is enough to feel directional without exploding cost.
                vec3 toBolt = closest - p;
                float distToBolt = max(1e-3, length(toBolt));
                vec3 boltDir = toBolt / distToBolt;
                float marchStep = min(0.35, distToBolt / 3.5);
                float occlusion = 0.0;
                for (int li = 1; li <= 3; li++) {
                    vec3 lp = p + boltDir * marchStep * float(li);
                    occlusion += max(0.0, scene(lp, true)) * marchStep;
                }
                float boltTransmittance = exp(-occlusion * u_absorption * 0.7);

                // Combine: distance falloff × shadowing × intensity.
                boltEnergy += totalTransmittance * density
                            * boltFalloff * boltTransmittance
                            * u_lightning;
            }

            // Early-out: once the view ray is essentially blocked, further
            // samples can't contribute much.
            if (totalTransmittance < 0.005) break;
        }
        depth += MARCH_SIZE;
        p = ro + depth * rd;
    }

    // Alpha for compositing against the sky. The article doesn't model a
    // separate "how much of the pixel is cloud" channel — implicitly,
    // anywhere we accumulated light is "cloud", anywhere we didn't is sky.
    // Use the inverse of accumulated transmittance as alpha.
    float alpha = clamp(1.0 - totalTransmittance, 0.0, 1.0);
    // Pack bolt energy into the green channel so the compositor can add it.
    return vec4(lightEnergy, boltEnergy, lightEnergy, alpha);
}

// 3-stop sky gradient (horizon → mid → top). Sunset palettes need the mid
// band — a 2-stop gradient washes out the magenta/purple transitions.
vec3 skyColor(vec3 rd) {
    float t = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 sky;
    if (t < 0.5) {
        sky = mix(u_sky_horizon, u_sky_mid, smoothstep(0.0, 0.5, t));
    } else {
        sky = mix(u_sky_mid, u_sky_top, smoothstep(0.5, 1.0, t));
    }
    // Solar disc + halo (only visible when sun is above horizon).
    float sun = pow(max(0.0, dot(rd, normalize(u_sun_dir))), 280.0);
    sky += u_sun_color * sun * 1.1;
    float halo = pow(max(0.0, dot(rd, normalize(u_sun_dir))), 8.0);
    sky += u_sun_color * halo * 0.08;
    return sky;
}

void main() {
    // Flip Y so the FBO is already oriented for QImage (top-left origin).
    // OpenGL's framebuffer origin is bottom-left, QImage's is top-left, so
    // historically we did a QImage.mirrored(False, True) on the CPU each
    // frame — a full-FBO copy. Flipping the sample point here makes that
    // copy unnecessary. Only `uv` (camera ray) cares about Y orientation;
    // `gl_FragCoord` used later for dither is unaffected because the noise
    // pattern is per-pixel and aperiodic — swapping which pixel gets which
    // dither value is invisible.
    vec2 fragXY = vec2(gl_FragCoord.x, u_resolution.y - gl_FragCoord.y);
    vec2 uv = (fragXY - 0.5 * u_resolution) / u_resolution.y;

    vec3 forward = normalize(u_cam_target - u_cam_pos);
    vec3 right   = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up      = cross(right, forward);

    vec3 ro = u_cam_pos;
    vec3 rd = normalize(uv.x * right + uv.y * up + 1.4 * forward);

    float dither = ign(gl_FragCoord.xy + float(u_frame % 64) * 5.588238);

    vec4 cloud = raymarch(ro, rd, dither);
    vec3 sky = skyColor(rd);

    // Storm darkens the sky toward an extremely deep moonlit blue. At full
    // storm intensity we also paint a soft moon disc + halo into the sky
    // — the only light source in the otherwise-black scene.
    sky = mix(sky, u_storm_sky, u_storm * 0.92);
    if (u_storm > 0.05 && u_moon_strength > 0.001) {
        vec3 moonDir = normalize(u_moon_dir);
        float md = max(0.0, dot(rd, moonDir));
        float moonDisc = pow(md, 480.0);     // tight bright disc
        float moonHalo = pow(md, 14.0);      // wider soft halo
        // Cool pale blue moonlight.
        vec3 moonCol = vec3(0.92, 0.95, 1.05);
        sky += moonCol * moonDisc * u_storm * u_moon_strength * 1.4;
        sky += moonCol * moonHalo * u_storm * u_moon_strength * 0.10;
    }

    // Tint the cloud using the time-of-day palette uniforms.
    //   - cloud.r is raw accumulated light energy (no longer bounded to 0..1).
    //     Pass through a soft saturating curve so highlights bloom into the
    //     warm sun color while shadows keep the cool tint.
    //   - The pow(lit, ...) terms drive the dramatic light-to-dark contrast
    //     across the cloud surface — high exponents on the sun-color mix
    //     mean only the brightest pixels get the rim highlight, exactly the
    //     "silver lining" look in the reference photograph.
    float lit = 1.0 - exp(-cloud.r * 2.4);            // soft saturate ~[0,1]
    vec3 cloudCol = mix(u_cloud_shadow, u_cloud_light, lit);
    cloudCol = mix(cloudCol, u_sun_color, pow(lit * (1.0 - u_storm * 0.8), 3.5) * 0.85);

    // Storm darkening: progressively push the cloud toward the deep night
    // palette as `u_storm` rises. Uses uniforms so the moonlit-storm look
    // can extend past the sunset palette without hardcoded fallbacks.
    vec3 stormCloud = mix(u_storm_cloud_dark, u_storm_cloud_lit, lit);
    cloudCol = mix(cloudCol, stormCloud, u_storm);

    // Additive lightning emission inside the cloud body. cloud.g holds
    // the accumulated bolt energy from the raymarch — anywhere the ray
    // passed through dense regions near the strike point glows.
    cloudCol += u_lightning_color * cloud.g * 3.5;

    vec3 col = mix(sky, cloudCol, cloud.a);

    // Visible bolt streak. Compute the *continuous* minimum distance from
    // the view ray to the bolt polyline. The previous version sampled
    // discrete spheres along the path, which read as "beads on a string"
    // because each sample's Gaussian had no overlap with its neighbour.
    // The new approach uses true ray-segment minimum distance per branch,
    // yielding a single connected line per branch — the dendritic
    // structure now reads correctly.
    if (u_lightning > 0.001) {
        float rayT;
        float minDist = boltDistanceToRay(ro, rd, rayT);

        // Two-zone falloff: tight white-hot core + wider color-tinted glow.
        float coreR = 0.020;
        float glowR = 0.18;
        float coreI = exp(-pow(minDist / coreR, 2.0));
        float glowI = exp(-pow(minDist / glowR, 2.0)) * 0.35;

        // Reject everything if the closest approach is behind the camera.
        if (rayT < 0.0) {
            coreI = 0.0;
            glowI = 0.0;
        }

        // Cloud occlusion: streak shows through wisps, hidden behind cores.
        float boltVisibility = mix(1.0, (1.0 - cloud.a), 0.7);

        vec3 streak = u_lightning_color * (coreI + glowI) * u_lightning * boltVisibility;
        // White-hot core regardless of the strike's overall color tint.
        streak += vec3(1.0, 1.0, 1.0) * coreI * u_lightning * boltVisibility * 0.6;
        col += streak * 2.4;
    }

    // Sky-wide flash: a real strike doesn't uniformly brighten the whole
    // sky — it lights up the region *near* the bolt much more than the
    // far side. We use the bolt's first joint (main-channel origin) as
    // the reference position; for typical strikes this is a fine proxy.
    if (u_lightning > 0.001 && u_storm > 0.05 && u_bolt_count > 0) {
        vec3 boltMid = u_bolt_joints[0];
        // If the polyline has enough joints, average with mid-channel for
        // a slightly better centroid.
        if (u_bolt_count > 4) {
            boltMid = mix(u_bolt_joints[0], u_bolt_joints[u_bolt_count / 2], 0.5);
        }
        vec3 toBolt = normalize(boltMid - ro);
        float align = max(0.0, dot(rd, toBolt));
        float skyMask = pow(align, 1.8);
        col += u_lightning_color * u_lightning * skyMask * 0.22 * u_storm;
    }

    // Tonemap: slightly stronger than before so the new HDR-ish output
    // doesn't blow out. Vignette only brightens corners (mix(0.92,1)),
    // doesn't darken — the previous 0.82 multiplier was flattening
    // edge contrast.
    col = col / (col + vec3(0.18)) * 1.18;
    float vig = smoothstep(1.4, 0.4, length(uv));
    col *= mix(0.92, 1.0, vig);

    frag = vec4(col, 1.0);
}
"""


# ─────────────────────────────────────────────
#  Time-of-day palette
# ─────────────────────────────────────────────
#
# `time_of_day` ∈ [0, 1]:
#   0.00 = pre-dawn   (true night, sun well below horizon)
#   0.18 = sunrise    (pink/peach horizon, lavender sky — matches the ref!)
#   0.50 = noon       (sun overhead, clean blue, white sun)
#   0.82 = sunset     (magenta/orange horizon, purple top)
#   1.00 = post-dusk  (true night, colder than pre-dawn)
#
# Hand-picked keyframes interpolated with smoothstep — feels nicer than
# computing colors from physics.

def _lerp(a, b, t):
    """Linear interpolate. Works on flat numeric tuples *and* nested tuples
    (so we can interpolate a whole palette of color tuples in one call)."""
    out = []
    for ai, bi in zip(a, b):
        if isinstance(ai, tuple):
            out.append(_lerp(ai, bi, t))
        else:
            out.append(ai + (bi - ai) * t)
    return tuple(out)


def _interp_keyframes(t, keyframes):
    """keyframes = [(t, value_tuple), ...] in ascending t order."""
    if t <= keyframes[0][0]:
        return keyframes[0][1]
    if t >= keyframes[-1][0]:
        return keyframes[-1][1]
    for i in range(len(keyframes) - 1):
        t0, v0 = keyframes[i]
        t1, v1 = keyframes[i + 1]
        if t0 <= t <= t1:
            local = (t - t0) / (t1 - t0)
            local = local * local * (3.0 - 2.0 * local)  # smoothstep
            return _lerp(v0, v1, local)
    return keyframes[-1][1]


# Each palette = (sky_horizon, sky_mid, sky_top, sun_color,
#                  cloud_shadow, cloud_light)
_PALETTES = [
    # 0.00 — pre-dawn: true night, slightly warming at horizon as sun
    # approaches from below. Cloud shadow is near-black; cloud_light is a
    # cool dim grey-blue so silhouettes barely catch any sky light.
    (0.00,
     (0.10, 0.08, 0.18),    # sky_horizon  — deep indigo with a hint of warmth
     (0.05, 0.04, 0.14),    # sky_mid
     (0.02, 0.02, 0.08),    # sky_top      — nearly black overhead
     (0.20, 0.15, 0.25),    # sun_color    — sun is below horizon, barely glows
     (0.05, 0.05, 0.10),    # cloud_shadow — black
     (0.22, 0.22, 0.32)),   # cloud_light  — faint cold grey-blue

    # 0.18 — sunrise: pink horizon, lavender sky — the reference vibe
    (0.18,
     (1.00, 0.75, 0.78),
     (0.78, 0.62, 0.92),
     (0.45, 0.45, 0.92),
     (1.00, 0.78, 0.62),
     (0.55, 0.45, 0.78),
     (1.00, 0.88, 0.95)),

    # 0.35 — morning: warm but blue-shifting
    (0.35,
     (0.95, 0.88, 0.82),
     (0.65, 0.78, 0.95),
     (0.30, 0.55, 0.92),
     (1.00, 0.92, 0.78),
     (0.55, 0.62, 0.80),
     (1.00, 0.98, 0.92)),

    # 0.50 — noon: clean blue, white sun
    (0.50,
     (0.85, 0.90, 0.95),
     (0.55, 0.75, 0.95),
     (0.20, 0.45, 0.85),
     (1.00, 0.97, 0.92),
     (0.50, 0.62, 0.80),
     (1.00, 1.00, 0.98)),

    # 0.65 — afternoon: warming back up
    (0.65,
     (0.98, 0.85, 0.72),
     (0.70, 0.72, 0.92),
     (0.30, 0.50, 0.88),
     (1.00, 0.86, 0.65),
     (0.55, 0.55, 0.78),
     (1.00, 0.94, 0.85)),

    # 0.82 — sunset: strong magenta/orange
    (0.82,
     (1.00, 0.55, 0.45),
     (0.85, 0.50, 0.78),
     (0.40, 0.30, 0.78),
     (1.00, 0.62, 0.40),
     (0.50, 0.35, 0.62),
     (1.00, 0.78, 0.72)),

    # 1.00 — post-dusk: true night, colder than pre-dawn (no rising sun).
    # All channels pushed down hard so the scene reads as moonlit-dark.
    (1.00,
     (0.07, 0.06, 0.16),    # sky_horizon  — cold deep blue
     (0.04, 0.03, 0.12),    # sky_mid
     (0.01, 0.01, 0.06),    # sky_top      — black overhead
     (0.18, 0.15, 0.28),    # sun_color    — sun gone, faint residual glow
     (0.04, 0.04, 0.09),    # cloud_shadow — black
     (0.20, 0.22, 0.32)),   # cloud_light  — cool moonlit silver-blue
]


# Pre-split into (t, value_tuple) form once at module load. palette_for_time
# is called every frame *and* up to 100×/frame in the HUD gradient strip pre-
# render, so even this tiny list-comprehension cost compounds visibly.
_PALETTE_KEYFRAMES = [(p[0], p[1:]) for p in _PALETTES]


def palette_for_time(t):
    """Return (sky_horizon, sky_mid, sky_top, sun_color, cloud_shadow, cloud_light)."""
    if t <= 0.0:
        return _PALETTE_KEYFRAMES[0][1]
    if t >= 1.0:
        return _PALETTE_KEYFRAMES[-1][1]
    return _interp_keyframes(t, _PALETTE_KEYFRAMES)


def sun_dir_for_time(t):
    """
    Returns (pitch, yaw) for the sun.
      t = 0.0 → sun far below horizon, east-ish
      t = 0.5 → sun overhead
      t = 1.0 → sun far below horizon, west-ish
    """
    yaw = math.radians(-10.0 + 200.0 * t)
    pitch = math.radians(-12.0 + math.sin(t * math.pi) * 82.0)
    return pitch, yaw


# ─────────────────────────────────────────────
#  Optional hand tracking (MediaPipe + OpenCV)
# ─────────────────────────────────────────────
#
# Runs on a background thread so it never blocks the render loop. Writes
# the latest hand-x (0..1, left→right of the *mirrored* camera frame) to
# self.value. If MediaPipe or a webcam are unavailable, the thread sets
# self.error and exits silently.

class HandTracker:
    """
    Background webcam hand tracker.
    self.value      — latest hand x in [0,1], or None if no hand seen yet
    self.is_fist    — True when the user is currently making a fist
    self.latest_frame — most recent webcam frame (bytes, RGB, downsized) for
                      optional preview rendering. None until the first frame
                      arrives. Frame is ~160px wide; the camera-feed HUD
                      blits it directly.
    self.frame_size — (w, h) of latest_frame, or None
    self.enabled    — True while the worker is running successfully
    self.error      — str if startup failed, else None
    """

    def __init__(self):
        self.value = None
        self.is_fist = False
        self.latest_frame = None
        self.frame_size = None
        self.enabled = False
        self.error = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._thread = None

    @staticmethod
    def _detect_fist(lms):
        """
        Robust fist detection from 21 MediaPipe hand landmarks.

        For each of the four fingers (index, middle, ring, pinky), a curled
        finger has its tip closer to the wrist than its PIP joint is. The
        ratio test is naturally scale-invariant so it works at any distance.

        Thumb is excluded — its anatomy means even a closed fist often has
        the thumb tip extended slightly past its IP joint.
        """
        lm = lms.landmark
        wrist = lm[0]

        def dist_sq_to_wrist(p):
            return ((p.x - wrist.x) ** 2 +
                    (p.y - wrist.y) ** 2 +
                    (p.z - wrist.z) ** 2)

        # (tip_idx, pip_idx) for index, middle, ring, pinky.
        finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
        curled = 0
        for tip_i, pip_i in finger_pairs:
            # Curled = tip is closer to wrist than the PIP joint (with a
            # small margin for stability).
            if dist_sq_to_wrist(lm[tip_i]) < dist_sq_to_wrist(lm[pip_i]) * 0.95:
                curled += 1
        return curled >= 3  # 3-of-4 tolerates one finger lagging

    def _run(self):
        try:
            import cv2  # type: ignore
            import mediapipe as mp  # type: ignore
        except ImportError as e:
            self.error = f"hand tracking unavailable: {e.name} not installed"
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.error = "hand tracking unavailable: no webcam"
            return

        hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.enabled = True
        self.error = None

        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                # Mirror: moving your hand right = palette moves right.
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                if result.multi_hand_landmarks:
                    lms = result.multi_hand_landmarks[0]
                    # Wrist (landmark 0) is the most stable anchor.
                    x = lms.landmark[0].x
                    self.value = max(0.0, min(1.0, float(x)))
                    self.is_fist = self._detect_fist(lms)
                else:
                    # No hand visible — drop fist state so the storm decays.
                    # (We keep self.value so palette doesn't snap.)
                    self.is_fist = False

                # Stash a downsized RGB frame for the optional preview HUD.
                # The widget reads this lock-free; tearing is harmless (worst
                # case is one frame showing half-old/half-new pixels).
                h0, w0 = rgb.shape[:2]
                target_w = 160
                if w0 > target_w:
                    scale = target_w / float(w0)
                    new_w = target_w
                    new_h = max(1, int(h0 * scale))
                    small = cv2.resize(rgb, (new_w, new_h),
                                       interpolation=cv2.INTER_AREA)
                else:
                    small = rgb
                self.frame_size = (small.shape[1], small.shape[0])
                # tobytes copy is small (~25 KB) and safer than sharing
                # the numpy buffer across threads.
                self.latest_frame = small.tobytes()
        finally:
            cap.release()
            hands.close()
            self.enabled = False


# ─────────────────────────────────────────────
#  The widget
# ─────────────────────────────────────────────

class CloudRaymarchWidget(QWidget):
    """
    Volumetric cloud raymarcher.
      drag           — orbit camera
      wheel          — zoom
      ← / →          — step time of day
      space          — toggle auto time progression
      h              — toggle hand tracking (if MediaPipe is installed)
      p              — toggle right-side parameter panel + bottom hint strip
      c              — toggle webcam picture-in-picture
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setMouseTracking(True)

        # Camera
        self.cam_distance = 7.0
        self.cam_yaw = 0.6
        self.cam_pitch = 0.15
        self.target = (0.0, 0.0, 0.0)

        # Time of day controls everything light-related.
        self.time_of_day = 0.18    # start at sunrise — matches the reference
        self.auto_time = False
        self.auto_time_rate = 0.04 # cycles per second when auto

        # Cloud shape parameters — tuned for the "defined" look.
        self.absorption = 1.8          # light-march absorption
        self.aniso = 0.72              # forward scattering — silver lining
        self.cloud_radius = 1.7
        self.cloud_radius_base = 1.7   # storm-free baseline
        self.density_scale = 1.4
        self.density_scale_base = 1.4  # storm-free baseline
        self.density_threshold = 0.42  # the silhouette knob
        self.edge_sharpness = 1.5
        self.warp_strength = 0.40      # domain-warp twist (tapered by envelope)
        self.detail_sharpness = 1.30   # edge-only power curve — wispy fingers

        # Grab / storm state.
        #   grab_charge      — 0..1, builds up while the fist is held, decays
        #                      when released. Drives cloud size + darkness.
        #   fist_hold_time   — seconds the current fist has been held (resets
        #                      on release). Used to trigger lightning at >5s.
        #   storm            — smoothed version of grab_charge passed to the
        #                      shader (extra smoothing kills any popping).
        #   lightning_*      — current strike state.
        #   next_strike_at   — seconds-since-storm-onset for the next bolt.
        self.grab_charge = 0.0
        self.fist_hold_time = 0.0
        self.storm = 0.0
        self.envelope_stretch = (1.0, 1.0, 1.0)
        self._storm_cam_distance = None  # filled each frame
        self.lightning_intensity = 0.0
        self.lightning_color = (1.0, 1.0, 1.0)
        self.lightning_radius = 0.35    # glow tube radius

        # Polyline joints + branch IDs for the current strike. The shader
        # has a fixed MAX_BOLT_JOINTS = 96; we pad unused entries with
        # zeros and pass u_bolt_count to mark the active range.
        self.MAX_BOLT_JOINTS = 96
        self.bolt_joints = [(0.0, 0.0, 0.0)] * self.MAX_BOLT_JOINTS
        self.bolt_branch_id = [0] * self.MAX_BOLT_JOINTS
        self.bolt_count = 0
        # Set True whenever bolt_joints/branch_id are rewritten by a new
        # strike, cleared after the next GPU upload. Avoids re-pushing a
        # 96-element array every frame while a strike's flicker plays out.
        self._bolt_dirty = False

        # Flicker envelope: list of (time_offset, peak_intensity) pulses for
        # the *current* strike. Computed once when the strike fires and
        # evaluated each frame against `_strike_started`.
        self._strike_pulses = []
        self._strike_started = 0.0
        self._strike_duration = 0.0
        self._next_strike_at = None    # set when storm crosses the 5s threshold
        self._storm_elapsed = 0.0      # seconds since fist_hold_time crossed 5s

        # Hand tracking
        self.hand_tracker = HandTracker()
        self.hand_active = False  # user-toggled

        # HUD toggles
        #   show_panel   — right-side parameter panel + bottom hint pill
        #                  + the time-of-day slider strip (P key — all
        #                  "informational chrome", one switch).
        #   show_camera  — picture-in-picture webcam feed (C key)
        # Both default off-ish: panel visible, camera hidden until requested.
        self.show_panel = True
        self.show_camera = False
        # Cached pre-rendered gradient strip for the time-of-day slider.
        # Built lazily on first paint; the palette is constant so this lives
        # for the widget's lifetime.
        self._gradient_img = None

        # Mouse drag
        self._drag_last = None

        # GL (lazy)
        self._gl_ready = False
        self.ctx = None
        self.fbo = None
        self.prog = None
        self.vao = None
        self._fbo_w = 0
        self._fbo_h = 0
        self._frame_image = None
        # Persistent readback buffer, sized to match the FBO. Allocated in
        # _resize_fbo so it stays in sync with the framebuffer dimensions.
        self._frame_bytes = bytearray(0)

        # Timing
        self.frame_count = 0
        self.start_time = time.perf_counter()
        self.last_time = None
        self._fps_smoothed = 0.0
        self._fps_accum = 0.0
        self._fps_frames = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(16)

    # ── GL init ────────────────────────────────────────────
    def _ensure_gl(self):
        if self._gl_ready:
            return
        self._gl_ready = True

        self.ctx = moderngl.create_context(standalone=True)
        self.prog = self.ctx.program(
            vertex_shader=VERT_SHADER, fragment_shader=FRAG_SHADER
        )

        verts = np.array([
            -1.0, -1.0,
             3.0, -1.0,
            -1.0,  3.0,
        ], dtype='f4')
        vbo = self.ctx.buffer(verts.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(vbo, '2f', 'in_pos')])

        # Constant uniforms — set once here so the per-frame uniform pass
        # only touches values that actually change. The storm palette and
        # moon direction are derived from hardcoded constants in the old
        # _render_gl_frame; pulling them out shaves ~5 uniform writes per
        # frame and makes the hot path cleaner.
        u = self.prog
        u['u_storm_sky'].value        = (0.018, 0.025, 0.055)
        u['u_storm_cloud_dark'].value = (0.015, 0.020, 0.040)
        u['u_storm_cloud_lit'].value  = (0.42,  0.50,  0.65)
        moon_pitch = math.radians(55.0)
        moon_yaw   = math.radians(20.0)
        u['u_moon_dir'].value = (
            math.cos(moon_pitch) * math.sin(moon_yaw),
            math.sin(moon_pitch),
            math.cos(moon_pitch) * math.cos(moon_yaw),
        )

        self._resize_fbo(max(self.width(), 320), max(self.height(), 240))
        self.last_time = time.perf_counter()
        self.timer.start()

    def _resize_fbo(self, w, h):
        rw, rh = max(2, w // 2), max(2, h // 2)
        if rw == self._fbo_w and rh == self._fbo_h and self.fbo:
            return
        if self.fbo:
            self.fbo.release()
        self._fbo_w, self._fbo_h = rw, rh
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((rw, rh), 4)],
        )
        # Keep the readback buffer in sync. Drop the QImage too — it points
        # into the old buffer and would crash if drawn during the next paint.
        self._frame_bytes = bytearray(rw * rh * 4)
        self._frame_image = None

    # ── Frame loop ─────────────────────────────────────────
    def _tick(self):
        now = time.perf_counter()
        dt = min(now - self.last_time, 0.05) if self.last_time else 0.016
        self.last_time = now
        self.frame_count += 1

        # FPS, smoothed
        self._fps_accum += dt
        self._fps_frames += 1
        if self._fps_accum >= 0.5:
            inst = self._fps_frames / self._fps_accum
            self._fps_smoothed = inst if self._fps_smoothed == 0 else \
                self._fps_smoothed * 0.7 + inst * 0.3
            self._fps_accum = 0.0
            self._fps_frames = 0

        # ── Fist / grab dynamics ──────────────────────────────────────
        # The fist gesture is a *separate* control from the hand-x→time-
        # of-day mapping. Charge accumulates the longer the user holds a
        # fist; it decays slowly when the hand opens, so brief glitches in
        # tracking don't dissolve the storm.
        is_fisting = (self.hand_active and self.hand_tracker.enabled
                      and self.hand_tracker.is_fist)
        if is_fisting:
            self.fist_hold_time += dt
            # Charge takes ~4 s of solid fisting to fully saturate.
            self.grab_charge = min(1.0, self.grab_charge + dt / 4.0)
        else:
            self.fist_hold_time = 0.0
            # Slow decay so the sky doesn't snap back on a tracking blip,
            # but storms clear within ~3 s of opening the hand.
            self.grab_charge = max(0.0, self.grab_charge - dt / 3.0)

        # Smooth `storm` toward `grab_charge` so per-frame jitter on the
        # tracker doesn't show up as visible flicker in cloud darkness.
        self.storm += (self.grab_charge - self.storm) * min(1.0, dt * 3.0)

        # Drive cloud size + density from grab charge. Both grow with the
        # storm — bigger angrier cloud the longer you grab.
        #   - Uniform radius growth: cloud roughly triples in scale.
        #   - Lateral stretch (envelope_stretch): on top of radius growth,
        #     stretch x/z another ~2× so the cloud spreads horizontally
        #     across the sky like a real anvil/shelf cloud — y barely grows
        #     because real storm cells flatten at the tropopause.
        #   - Density: thicker interior for the opaque-thundercloud look.
        self.cloud_radius = self.cloud_radius_base * (1.0 + self.storm * 2.0)
        self.envelope_stretch = (
            1.0 + self.storm * 2.2,   # x — spread across sky
            1.0 + self.storm * 0.25,  # y — barely taller
            1.0 + self.storm * 2.2,   # z — spread in depth too
        )
        self.density_scale = self.density_scale_base * (1.0 + self.storm * 1.0)

        # Camera pulls back during the storm so the user actually sees the
        # spread — otherwise a maxed-out storm would have us inside the
        # cloud. We modulate the *effective* distance only; the user's
        # mouse-wheel zoom value is preserved so opening their hand
        # returns to wherever they had it.
        self._storm_cam_distance = self.cam_distance * (1.0 + self.storm * 1.4)

        # ── Lightning scheduling ──────────────────────────────────────
        # Lightning only triggers after the user has held a fist for >5 s
        # *and* the storm is actually visible. Bolts fire on a randomized
        # interval (1.5–4 s) for as long as the fist is held past 5 s.
        if self.fist_hold_time > 5.0 and self.storm > 0.4:
            if self._next_strike_at is None:
                # Just crossed the threshold — fire one bolt almost
                # immediately for the "you've earned a lightning storm" beat.
                self._next_strike_at = self._storm_elapsed + 0.3
            self._storm_elapsed += dt
            if self._storm_elapsed >= self._next_strike_at:
                self._fire_lightning_bolt(random)
                self._next_strike_at = self._storm_elapsed + random.uniform(1.5, 4.0)
        else:
            # Reset scheduling so the next storm starts fresh.
            self._next_strike_at = None
            self._storm_elapsed = 0.0

        # Evaluate the multi-pulse flicker envelope. Each pulse is a sharp
        # asymmetric spike: very fast rise (~3 ms feel, since we're sampling
        # at ~60 Hz this looks instant) and a short decay (~40 ms). Sum of
        # all active pulses gives the characteristic stroboscopic flicker.
        if self._strike_pulses:
            since = time.perf_counter() - self._strike_started
            total = 0.0
            for offset, amp in self._strike_pulses:
                dt_p = since - offset
                if dt_p < 0.0:
                    continue
                # Asymmetric pulse: instant rise, exponential decay with a
                # ~40 ms time constant. The clamp keeps it from going
                # negative on long-past pulses.
                contribution = amp * math.exp(-dt_p / 0.04)
                if contribution > 0.001:
                    total += contribution
            self.lightning_intensity = total
            # Clear the strike once it's fully decayed.
            if since > self._strike_duration and total < 0.005:
                self._strike_pulses = []
                self.lightning_intensity = 0.0
        else:
            self.lightning_intensity = 0.0

        # Drive time_of_day. Priority: hand control → auto → none.
        # Hand-x → time-of-day stays active even while fisting; the fist
        # gesture controls storm intensity (vertical-ish state), while
        # left/right hand motion independently scrubs through the
        # time-of-day palette. The two channels don't interfere — you can
        # summon a sunrise storm or a midnight storm by holding the fist
        # at different horizontal positions.
        if self.hand_active and self.hand_tracker.value is not None:
            # Smooth toward the hand value — micro-jitter would cause palette
            # shimmer otherwise.
            target = self.hand_tracker.value
            self.time_of_day += (target - self.time_of_day) * min(1.0, dt * 4.0)
        elif self.auto_time:
            self.time_of_day = (self.time_of_day + self.auto_time_rate * dt) % 1.0

        self._render_gl_frame()
        self.update()

    def _fire_lightning_bolt(self, rng):
        """Generate a dendritic lightning strike and start the flash envelope.

        Path generation uses recursive midpoint displacement (a fractal
        subdivision that produces fractional-Brownian-motion-like paths).
        The main channel runs from a point inside the cloud down through
        the cloud base; 2–4 side branches fork off at random points along
        the main channel and run shorter sub-channels. All joints flatten
        into a single uniform array with branch IDs so the shader can
        avoid drawing connector lines between unrelated branches.

        Strike personality (narrow / sheet / branched) controls the glow
        radius, intensity, duration, and number of branches.
        """
        sx, sy, sz = self.envelope_stretch

        # ── Main-channel origin: random spot in the upper-mid cloud ──
        r = self.cloud_radius * (0.2 + rng.random() * 0.6)
        theta = rng.uniform(0.0, 2.0 * math.pi)
        phi = rng.uniform(0.15, 0.55) * math.pi   # upper-hemisphere bias
        ox = r * math.sin(phi) * math.cos(theta) * sx
        oy = r * math.cos(phi) * 0.7 * sy
        oz = r * math.sin(phi) * math.sin(theta) * sz

        # ── Main-channel tip: down and slightly outward ──
        bolt_len = self.cloud_radius * rng.uniform(0.8, 1.6)
        drift_theta = theta + rng.uniform(-0.6, 0.6)
        dx = math.cos(drift_theta) * bolt_len * rng.uniform(0.15, 0.45)
        dy = -bolt_len  # mostly downward
        dz = math.sin(drift_theta) * bolt_len * rng.uniform(0.15, 0.45)
        ex, ey, ez = ox + dx, oy + dy, oz + dz

        # ── Strike personality (drives radius, intensity, n_branches) ──
        roll = rng.random()
        if roll < 0.50:
            self.lightning_radius = rng.uniform(0.18, 0.30)
            peak_intensity = rng.uniform(1.0, 1.4)
            self._strike_duration = rng.uniform(0.20, 0.35)
            n_branches = rng.randint(2, 4)
        elif roll < 0.80:
            self.lightning_radius = rng.uniform(0.55, 0.85)
            peak_intensity = rng.uniform(0.7, 0.95)
            self._strike_duration = rng.uniform(0.12, 0.22)
            n_branches = rng.randint(0, 2)
        else:
            self.lightning_radius = rng.uniform(0.28, 0.42)
            peak_intensity = rng.uniform(1.2, 1.6)
            self._strike_duration = rng.uniform(0.30, 0.55)
            n_branches = rng.randint(3, 5)

        # ── Build the dendrite ──
        # First the main channel as a midpoint-displaced fractal polyline,
        # then side branches forking off from interior joints.
        max_joints = self.MAX_BOLT_JOINTS
        # Reserve enough budget so branches don't overflow the array.
        main_levels = 4         # 2^4 + 1 = 17 joints on the main channel
        branch_levels = 3       # 2^3 + 1 = 9 joints per branch
        main_jitter = bolt_len * 0.08   # initial perpendicular jitter for main
        branch_jitter_scale = 0.6

        main_path = self._fractal_polyline(
            (ox, oy, oz), (ex, ey, ez),
            levels=main_levels, jitter=main_jitter, rng=rng,
        )

        joints = []
        branch_ids = []
        for jp in main_path:
            joints.append(jp)
            branch_ids.append(0)

        # ── Side branches ──
        # Fork from random main-path joints (excluding endpoints). Branch
        # length is 30–60% of the remaining distance along the main
        # channel from the fork point; direction is the main-channel
        # direction rotated by a sharp angle plus extra random vertical.
        for b in range(n_branches):
            if len(joints) + (2 ** branch_levels + 1) > max_joints:
                break
            # Fork point — somewhere in the upper-mid main path.
            fork_idx = rng.randint(2, max(2, len(main_path) - 3))
            fpx, fpy, fpz = main_path[fork_idx]

            # Main direction at the fork (approx, using next joint).
            nx, ny, nz = main_path[min(fork_idx + 1, len(main_path) - 1)]
            mdx, mdy, mdz = nx - fpx, ny - fpy, nz - fpz
            mlen = max(1e-6, math.sqrt(mdx * mdx + mdy * mdy + mdz * mdz))
            mdx, mdy, mdz = mdx / mlen, mdy / mlen, mdz / mlen

            # Branch direction: rotate ~30–70° away from main direction.
            # Pick a random perpendicular axis, build a rotation in that plane.
            # Cheap approximation: take a random unit vector, project out
            # the main-direction component, normalize → perpendicular.
            rax = rng.uniform(-1.0, 1.0)
            ray = rng.uniform(-1.0, 1.0)
            raz = rng.uniform(-1.0, 1.0)
            # Project: r' = r - (r·m) m
            d = rax * mdx + ray * mdy + raz * mdz
            px_, py_, pz_ = rax - d * mdx, ray - d * mdy, raz - d * mdz
            plen = max(1e-6, math.sqrt(px_ * px_ + py_ * py_ + pz_ * pz_))
            px_, py_, pz_ = px_ / plen, py_ / plen, pz_ / plen

            # Branch axis = mix(main, perp) — main-weighted so branches
            # still tend downward, perp-weighted enough to spread sideways.
            angle = math.radians(rng.uniform(30.0, 70.0))
            bx_ = math.cos(angle) * mdx + math.sin(angle) * px_
            by_ = math.cos(angle) * mdy + math.sin(angle) * py_
            bz_ = math.cos(angle) * mdz + math.sin(angle) * pz_

            branch_len = bolt_len * rng.uniform(0.25, 0.55)
            btx, bty, btz = (fpx + bx_ * branch_len,
                              fpy + by_ * branch_len,
                              fpz + bz_ * branch_len)

            branch_path = self._fractal_polyline(
                (fpx, fpy, fpz), (btx, bty, btz),
                levels=branch_levels,
                jitter=branch_len * 0.10 * branch_jitter_scale,
                rng=rng,
            )
            bid = b + 1
            for jp in branch_path:
                if len(joints) >= max_joints:
                    break
                joints.append(jp)
                branch_ids.append(bid)

        # Fill the fixed-size uniform arrays.
        self.bolt_count = len(joints)
        # Pad to MAX_BOLT_JOINTS — the shader respects u_bolt_count, but
        # ModernGL still wants the full array passed each frame.
        padded = joints + [(0.0, 0.0, 0.0)] * (max_joints - len(joints))
        padded_ids = branch_ids + [0] * (max_joints - len(branch_ids))
        self.bolt_joints = padded
        self.bolt_branch_id = padded_ids
        # Flag for the next render to push the new geometry; cleared there.
        self._bolt_dirty = True

        # ── Color tint ──
        if rng.random() < 0.20:
            self.lightning_color = (1.0, 0.92, 0.80)
        else:
            self.lightning_color = (0.85, 0.92, 1.0)

        # ── Flicker envelope (2–4 sub-pulses, fast-rise + exp decay) ──
        n_pulses = rng.randint(2, 4)
        self._strike_pulses = []
        t_offset = 0.0
        amp = peak_intensity
        for _ in range(n_pulses):
            self._strike_pulses.append((t_offset, amp))
            t_offset += rng.uniform(0.025, 0.085)
            amp *= rng.uniform(0.45, 0.75)

        self._strike_started = time.perf_counter()
        self.lightning_intensity = peak_intensity

    @staticmethod
    def _fractal_polyline(start, end, levels, jitter, rng):
        """Recursive midpoint-displacement subdivision.

        Returns a list of (x,y,z) joints from start to end with 2^levels + 1
        points. At each level we insert the midpoint of every existing
        segment, perturbed perpendicular to that segment by a random
        amount. The jitter halves at each level, giving fractional-
        Brownian-motion shape with characteristic 1/f-ish scaling — the
        same statistical structure that makes real lightning paths look
        the way they do.
        """
        path = [start, end]
        current_jitter = jitter
        for _ in range(levels):
            new_path = [path[0]]
            for i in range(len(path) - 1):
                a = path[i]
                b = path[i + 1]
                mx, my, mz = ((a[0] + b[0]) * 0.5,
                              (a[1] + b[1]) * 0.5,
                              (a[2] + b[2]) * 0.5)
                # Build a perpendicular vector to displace along.
                dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
                # Two arbitrary perpendiculars via cross with x then y axes.
                # Cross with (1,0,0): (0, -dz, dy). If degenerate, use (0,1,0).
                px, py, pz = 0.0, -dz, dy
                plen = math.sqrt(px * px + py * py + pz * pz)
                if plen < 1e-6:
                    px, py, pz = -dz, 0.0, dx
                    plen = math.sqrt(px * px + py * py + pz * pz)
                px, py, pz = px / plen, py / plen, pz / plen
                # Second perpendicular = (dx,dy,dz) × (px,py,pz), normalized.
                qx = dy * pz - dz * py
                qy = dz * px - dx * pz
                qz = dx * py - dy * px
                qlen = max(1e-6, math.sqrt(qx * qx + qy * qy + qz * qz))
                qx, qy, qz = qx / qlen, qy / qlen, qz / qlen
                # Random displacement in the perpendicular plane.
                u = (rng.random() - 0.5) * 2.0 * current_jitter
                v = (rng.random() - 0.5) * 2.0 * current_jitter
                mx += px * u + qx * v
                my += py * u + qy * v
                mz += pz * u + qz * v
                new_path.append((mx, my, mz))
                new_path.append(b)
            path = new_path
            current_jitter *= 0.5
        return path

    def _render_gl_frame(self):
        if not self._gl_ready:
            return
        w, h = max(self.width(), 320), max(self.height(), 240)
        self._resize_fbo(w, h)

        self.fbo.use()
        self.ctx.viewport = (0, 0, self._fbo_w, self._fbo_h)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        # Camera basis from orbit angles. Use the storm-modulated distance
        # so the view pulls back as the cloud grows — keeps the silhouette
        # fully visible at max storm.
        cam_d = self._storm_cam_distance if self._storm_cam_distance else self.cam_distance
        cx = cam_d * math.cos(self.cam_pitch) * math.sin(self.cam_yaw)
        cy = cam_d * math.sin(self.cam_pitch)
        cz = cam_d * math.cos(self.cam_pitch) * math.cos(self.cam_yaw)
        cam_pos = (cx, cy, cz)

        # Sun & palette from time of day.
        sun_pitch, sun_yaw = sun_dir_for_time(self.time_of_day)
        sx = math.cos(sun_pitch) * math.sin(sun_yaw)
        sy = math.sin(sun_pitch)
        sz = math.cos(sun_pitch) * math.cos(sun_yaw)

        (sky_horizon, sky_mid, sky_top,
         sun_color, cloud_shadow, cloud_light) = palette_for_time(self.time_of_day)

        # The storm-palette and moon-direction uniforms are constants now
        # set once in _ensure_gl. They never change between frames so we
        # don't touch them here — keeps the per-frame uniform pass focused
        # on values that actually vary with state.

        u = self.prog
        elapsed = time.perf_counter() - self.start_time
        u['u_resolution'].value         = (float(self._fbo_w), float(self._fbo_h))
        u['u_time'].value               = float(elapsed)
        u['u_frame'].value              = int(self.frame_count)
        u['u_cam_pos'].value            = cam_pos
        u['u_cam_target'].value         = self.target
        u['u_sun_dir'].value            = (sx, sy, sz)
        u['u_sun_color'].value          = sun_color
        u['u_sky_top'].value            = sky_top
        u['u_sky_mid'].value            = sky_mid
        u['u_sky_horizon'].value        = sky_horizon
        u['u_cloud_shadow'].value       = cloud_shadow
        u['u_cloud_light'].value        = cloud_light
        u['u_absorption'].value         = float(self.absorption)
        u['u_aniso'].value              = float(self.aniso)
        u['u_cloud_radius'].value       = float(self.cloud_radius)
        u['u_density_scale'].value      = float(self.density_scale)
        u['u_density_threshold'].value  = float(self.density_threshold)
        u['u_edge_sharpness'].value     = float(self.edge_sharpness)
        u['u_warp_strength'].value      = float(self.warp_strength)
        u['u_detail_sharpness'].value   = float(self.detail_sharpness)
        u['u_storm'].value              = float(self.storm)
        u['u_envelope_stretch'].value   = self.envelope_stretch
        u['u_moon_strength'].value      = float(self.storm)  # moon rises with storm
        u['u_lightning'].value          = float(self.lightning_intensity)
        u['u_lightning_color'].value    = self.lightning_color
        u['u_lightning_radius'].value   = float(self.lightning_radius)

        # Bolt joint array — only re-upload when a new strike has fired.
        # Joints don't move within a strike (the flicker is purely intensity),
        # so re-pushing the 96-tuple array every frame during an active
        # strike was wasted work. `_bolt_dirty` is set by _fire_lightning_bolt
        # and cleared here. Drop the per-element tuple()/list() wraps too:
        # joints already come out of the strike builder as tuples and
        # branch_id is already a list, so the copies were pure overhead.
        if self.lightning_intensity > 0.001 and self.bolt_count > 0:
            if self._bolt_dirty:
                # ModernGL dispatches per uniform type:
                #   - vec3[N] setter wants an iterable of 3-tuples
                #   - int[N] setter wants an iterable of plain ints
                u['u_bolt_joints'].value = self.bolt_joints
                u['u_bolt_branch_id'].value = self.bolt_branch_id
                self._bolt_dirty = False
            u['u_bolt_count'].value = self.bolt_count
        else:
            # Cheap path: when no strike is active, just zero the count
            # so the shader's loops bail out immediately. We can skip
            # updating the joint arrays — the shader won't read them.
            u['u_bolt_count'].value = 0

        self.vao.render(moderngl.TRIANGLES)

        # Read FBO into a persistent bytearray instead of allocating fresh
        # bytes every frame. At 640×400×4 = 1 MB/frame × 60 FPS that's 60
        # MB/s of allocation+free churn the old `.read()` path was causing.
        # Resizing the buffer happens in `_resize_fbo`, in lockstep with the
        # FBO itself, so by the time we get here `_frame_bytes` always has
        # the right size. The QImage wraps the bytearray without copying;
        # the buffer must stay alive on `self` for the lifetime of the
        # QImage, hence `self._frame_bytes` rather than a local.
        self.fbo.color_attachments[0].read_into(self._frame_bytes)
        self._frame_image = QImage(
            self._frame_bytes, self._fbo_w, self._fbo_h, self._fbo_w * 4,
            QImage.Format_RGBA8888
        )

    # ── Painting ───────────────────────────────────────────
    def paintEvent(self, event):
        self._ensure_gl()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        w, h = self.width(), self.height()

        if self._frame_image and not self._frame_image.isNull():
            painter.drawImage(
                self.rect(), self._frame_image, self._frame_image.rect()
            )
        else:
            painter.fillRect(0, 0, w, h, QColor(200, 220, 240))

        self._draw_hud(painter, w, h)
        painter.end()

    def _time_label(self):
        t = self.time_of_day
        if t < 0.08:   return "pre-dawn"
        if t < 0.25:   return "sunrise"
        if t < 0.42:   return "morning"
        if t < 0.58:   return "noon"
        if t < 0.75:   return "afternoon"
        if t < 0.92:   return "sunset"
        return "dusk"

    def _draw_hud(self, painter, w, h):
        # Storm status pill (top-center). Always visible regardless of the
        # P toggle — it's an *event* indicator, not a parameter panel.
        if self.grab_charge > 0.02 or self.fist_hold_time > 0.0:
            self._draw_storm_pill(painter, w, h)

        # The right-side parameter panel, the bottom hint pill, AND the
        # time-of-day slider strip are all "chrome" — toggle off with P
        # for a clean view of the cloud. The slider used to be considered
        # primary input, but moving it under the toggle gives a fully
        # unobstructed view when the user wants one; the cursor/wheel/key
        # bindings still drive time-of-day with the strip hidden.
        if self.show_panel:
            self._draw_param_panel(painter, w, h)
            self._draw_hint_pill(painter, w, h)
            self._draw_time_slider(painter, w, h)

        # Camera-feed picture-in-picture (top-right corner). Only renders
        # when the user has explicitly toggled it on AND the tracker has
        # delivered at least one frame.
        if self.show_camera:
            self._draw_camera_feed(painter, w, h)

        # If hand tracking failed, surface the reason just above the hint
        # pill area (even when the panel is hidden, so the user isn't
        # left wondering why H isn't working).
        if self.hand_active and self.hand_tracker.error:
            painter.setPen(QColor(255, 220, 220))
            painter.setFont(QFont("Helvetica", 8))
            painter.drawText(0, h - 56, w, 14, Qt.AlignCenter,
                             self.hand_tracker.error)

    def _draw_param_panel(self, painter, w, h):
        """Right-side parameter readout (toggled by P)."""
        box_w, box_h = 220, 210
        bx, by = w - box_w - 16, 16
        painter.setBrush(QColor(255, 255, 255, 215))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(bx, by, box_w, box_h, 12, 12)

        painter.setFont(QFont("monospace", 9))
        painter.setPen(QColor(60, 80, 110))
        sun_pitch_deg = math.degrees(sun_dir_for_time(self.time_of_day)[0])
        lines = [
            f"fps        {self._fps_smoothed:6.1f}",
            f"resolution {self._fbo_w}x{self._fbo_h}",
            f"absorption {self.absorption:6.2f}",
            f"aniso  (g) {self.aniso:+6.2f}",
            f"threshold  {self.density_threshold:6.2f}",
            f"density    {self.density_scale:6.2f}",
            f"warp       {self.warp_strength:6.2f}",
            f"detail     {self.detail_sharpness:6.2f}",
            f"time       {self.time_of_day:6.2f} {self._time_label()}",
            f"sun pitch  {sun_pitch_deg:6.1f}°",
            f"grab       {self.grab_charge:6.2f}  hold {self.fist_hold_time:4.1f}s",
        ]
        for i, line in enumerate(lines):
            painter.drawText(bx + 14, by + 22 + i * 15, line)

    def _draw_hint_pill(self, painter, w, h):
        """Bottom hint strip with keyboard shortcuts (toggled by P)."""
        hint_w = 640
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(20, 30, 50, 175))
        painter.drawRoundedRect(w // 2 - hint_w // 2, h - 36, hint_w, 22, 11, 11)
        painter.setPen(QColor(230, 240, 255))
        painter.setFont(QFont("Helvetica", 9))
        if self.hand_active:
            if self.hand_tracker.enabled:
                hand_state = "ON"
            elif self.hand_tracker.error:
                hand_state = "ERR"
            else:
                hand_state = "starting…"
        else:
            hand_state = "OFF"
        painter.drawText(w // 2 - hint_w // 2, h - 36, hint_w, 22,
                         Qt.AlignCenter,
                         f"drag · wheel · ←/→ · space · H:hand({hand_state}) "
                         f"· P:panel · C:camera  ✊ fist→storm (5s→lightning)")

    def _draw_camera_feed(self, painter, w, h):
        """Picture-in-picture webcam preview in the top-right corner.

        Uses the latest frame stashed by HandTracker (already mirrored and
        downsized to ~160px wide). If the panel is also showing, this sits
        below it.
        """
        frame = self.hand_tracker.latest_frame
        size = self.hand_tracker.frame_size
        if frame is None or size is None:
            # Tracker hasn't started or hasn't produced a frame yet —
            # show a placeholder so the user knows the toggle worked.
            pw, ph = 160, 90
            x = w - pw - 16
            y = 16
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 180))
            painter.drawRoundedRect(x, y, pw, ph, 8, 8)
            painter.setPen(QColor(200, 200, 220))
            painter.setFont(QFont("Helvetica", 9))
            msg = "camera off"
            if self.hand_active and not self.hand_tracker.error:
                msg = "starting camera…"
            elif self.hand_tracker.error:
                msg = "no camera"
            painter.drawText(x, y, pw, ph, Qt.AlignCenter, msg)
            return

        fw, fh = size
        # Position: top-right, below the parameter panel if it's open.
        x = w - fw - 16
        y = 16 + (220 if self.show_panel else 0)

        # Background frame to give it a soft border.
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 200))
        painter.drawRoundedRect(x - 4, y - 4, fw + 8, fh + 8, 6, 6)

        # Build the QImage from the raw bytes. The frame is RGB888.
        img = QImage(frame, fw, fh, fw * 3, QImage.Format_RGB888)
        painter.drawImage(x, y, img)

        # Label overlay so the user knows what they're looking at.
        painter.setPen(QColor(255, 255, 255, 200))
        painter.setFont(QFont("Helvetica", 8, QFont.Bold))
        tag = "CAM"
        if self.hand_tracker.is_fist:
            tag = "CAM ✊"
        painter.drawText(x + 6, y + 4, fw, 14, Qt.AlignLeft | Qt.AlignTop, tag)

    def _draw_storm_pill(self, painter, w, h):
        """Top-center status indicator for the fist→storm gesture."""
        # State: building (fist held <5s), active (lightning firing), decaying.
        lightning_armed = self.fist_hold_time > 5.0
        if lightning_armed:
            label = "⚡ STORM ACTIVE — LIGHTNING ⚡"
            bg = QColor(20, 10, 40, 220)
            fg = QColor(255, 240, 180)
        elif self.fist_hold_time > 0.0:
            remaining = max(0.0, 5.0 - self.fist_hold_time)
            label = f"STORM BUILDING — lightning in {remaining:.1f}s"
            bg = QColor(30, 30, 60, 200)
            fg = QColor(220, 230, 255)
        else:
            label = "storm dissipating…"
            bg = QColor(60, 60, 80, 160)
            fg = QColor(220, 220, 230)

        pill_w, pill_h = 360, 28
        px = w // 2 - pill_w // 2
        py = 18
        painter.setPen(Qt.NoPen)
        painter.setBrush(bg)
        painter.drawRoundedRect(px, py, pill_w, pill_h, 14, 14)

        # Progress fill — grows with grab_charge.
        if self.grab_charge > 0.0:
            fill_w = int((pill_w - 8) * self.grab_charge)
            fill_color = (QColor(180, 80, 220, 180) if lightning_armed
                          else QColor(120, 140, 220, 160))
            painter.setBrush(fill_color)
            painter.drawRoundedRect(px + 4, py + pill_h - 7, fill_w, 4, 2, 2)

        painter.setPen(fg)
        painter.setFont(QFont("Helvetica", 10, QFont.Bold))
        painter.drawText(px, py, pill_w, pill_h - 6, Qt.AlignCenter, label)

    def _build_time_slider_gradient(self):
        """Pre-render the time-of-day palette as a 1×N strip.

        The previous implementation re-sampled the palette and drew 100
        filled rectangles every paint event — 100 palette interpolations,
        100 QColor constructions, 100 setBrush + drawRect calls per frame
        for a strip that never changes. Caching this as a tiny QImage
        cuts the work to a single drawImage() with linear scaling.
        """
        n = 256  # smoother than the old 100-strip render at no real cost
        img = QImage(n, 1, QImage.Format_RGB888)
        for i in range(n):
            t = (i + 0.5) / n
            sky_horizon = palette_for_time(t)[0]
            img.setPixelColor(i, 0, QColor(
                int(sky_horizon[0] * 255),
                int(sky_horizon[1] * 255),
                int(sky_horizon[2] * 255),
            ))
        return img

    def _draw_time_slider(self, painter, w, h):
        """Visualize the time-of-day palette as a gradient strip with a cursor."""
        sx, sy = 16, h - 78
        sw, sh = w - 32, 22

        # Pre-rendered gradient: built once on first draw, blitted thereafter.
        if self._gradient_img is None:
            self._gradient_img = self._build_time_slider_gradient()
        # Stretch the 1-row gradient across the strip rect in one call.
        painter.drawImage(QRect(sx, sy, sw, sh), self._gradient_img,
                          self._gradient_img.rect())

        # Border
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
        painter.drawRoundedRect(sx, sy, sw, sh, 4, 4)

        # Cursor
        cx = int(sx + self.time_of_day * sw)
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(QPen(QColor(20, 30, 50), 2))
        painter.drawRect(cx - 3, sy - 3, 6, sh + 6)

        # Labels at the key times
        painter.setPen(QColor(255, 255, 255, 230))
        painter.setFont(QFont("Helvetica", 8, QFont.Bold))
        for label, t in [("sunrise", 0.18), ("noon", 0.5), ("sunset", 0.82)]:
            tx = int(sx + t * sw)
            painter.drawText(tx - 24, sy - 6, 48, 12, Qt.AlignCenter, label)

    # ── Interaction ────────────────────────────────────────
    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.LeftButton:
            self._drag_last = ev.position()

    def mouseMoveEvent(self, ev: QMouseEvent):
        if self._drag_last is None:
            return
        pos = ev.position()
        dx = pos.x() - self._drag_last.x()
        dy = pos.y() - self._drag_last.y()
        self._drag_last = pos
        self.cam_yaw -= dx * 0.007
        self.cam_pitch += dy * 0.007
        self.cam_pitch = max(-1.4, min(1.4, self.cam_pitch))

    def mouseReleaseEvent(self, ev: QMouseEvent):
        self._drag_last = None

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y() / 120.0
        self.cam_distance *= math.pow(0.9, delta)
        self.cam_distance = max(2.5, min(20.0, self.cam_distance))

    def keyPressEvent(self, ev):
        key = ev.key()
        if key == Qt.Key_Left:
            self.time_of_day = max(0.0, self.time_of_day - 0.02)
            self.auto_time = False
        elif key == Qt.Key_Right:
            self.time_of_day = min(1.0, self.time_of_day + 0.02)
            self.auto_time = False
        elif key == Qt.Key_Space:
            self.auto_time = not self.auto_time
            if self.auto_time:
                self.hand_active = False
        elif key == Qt.Key_H:
            self.hand_active = not self.hand_active
            if self.hand_active:
                self.hand_tracker.start()
                self.auto_time = False
        elif key == Qt.Key_P:
            # Toggle the parameter panel, bottom hint strip, and the
            # time-of-day slider together — all "informational chrome",
            # all on the same switch for a fully clean view of the cloud.
            self.show_panel = not self.show_panel
        elif key == Qt.Key_C:
            # Toggle the picture-in-picture webcam feed. Implicitly start
            # the tracker if it isn't running yet, otherwise we'd show an
            # empty placeholder forever. We don't enable hand_active —
            # the user can opt into hand control separately with H.
            self.show_camera = not self.show_camera
            if self.show_camera:
                self.hand_tracker.start()
        else:
            super().keyPressEvent(ev)

    def closeEvent(self, ev):
        self.hand_tracker.stop()
        super().closeEvent(ev)


# ─────────────────────────────────────────────
#  Bootstrap
# ─────────────────────────────────────────────

def _install_into_rio_scene(graphics_scene):
    """Build the widget and add it to an existing Rio scene."""

    for name in ('cloud_proxy', 'cloud_widget', 'cloud_container'):
        if name in globals():
            try:
                obj = globals()[name]
                if hasattr(obj, 'scene') and obj.scene() is graphics_scene:
                    graphics_scene.removeItem(obj)
            except Exception:
                pass
            globals().pop(name, None)

    container = QWidget()
    container.setFixedSize(1280, 800)
    container.setStyleSheet("background: #0b1020; border-radius: 14px;")
    lay = QVBoxLayout(container)
    lay.setContentsMargins(0, 0, 0, 0)

    widget = CloudRaymarchWidget()
    lay.addWidget(widget)

    proxy = graphics_scene.addWidget(container)

    views = graphics_scene.views()
    if views:
        v = views[0]
        vr = v.viewport().rect()
        sr = v.mapToScene(vr).boundingRect()
        proxy.setPos(sr.center().x() - 640, sr.center().y() - 400)
    else:
        proxy.setPos(0, 0)

    proxy.setFlag(QGraphicsItem.ItemIsMovable, True)
    widget.setFocus()

    return container, widget, proxy


def _run_standalone():
    import sys
    from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView
    from PySide6.QtGui import QPainter as _QP

    app = QApplication.instance() or QApplication(sys.argv)
    scene = QGraphicsScene()
    view = QGraphicsView(scene)
    view.setRenderHint(_QP.Antialiasing)
    view.setSceneRect(0, 0, 1300, 820)
    view.setFixedSize(1320, 840)
    _install_into_rio_scene(scene)
    view.show()
    sys.exit(app.exec())


if "graphics_scene" in globals() and globals()["graphics_scene"] is not None:
    cloud_container, cloud_widget, cloud_proxy = \
        _install_into_rio_scene(globals()["graphics_scene"])
elif __name__ == "__main__":
    _run_standalone()