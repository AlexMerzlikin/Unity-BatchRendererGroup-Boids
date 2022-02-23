using System.Runtime.CompilerServices;
using Unity.Mathematics;

namespace ThousandAnt.Boids {
    public static class TransformExtensions {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static quaternion Rotation(this in float4x4 m) {
            return new quaternion(m);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float3 Position(this in float4x4 m) {
            return m.c3.xyz;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float3 Forward(this in float4x4 m) {
            return m.c2.xyz;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static quaternion QuaternionBetween(this in float3 from, in float3 to) {
            var cross = math.cross(from, to);

            var w = math.sqrt(math.lengthsq(from) * math.lengthsq(to)) + math.dot(from, to);
            return new quaternion(new float4(cross, w));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float3 SeparationVector(float3 current, float3 other, float maxDist) {
            var diff   = current - other;
            var mag    = math.length(diff);
            var scalar = math.clamp(1 - mag / maxDist, 0, 1);

            return diff * (scalar / mag);
        }
    }
}
