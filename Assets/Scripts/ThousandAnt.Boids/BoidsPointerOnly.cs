using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace ThousandAnt.Boids {

    public unsafe class BoidsPointerOnly {

        [BurstCompile]
        public struct AverageCenterJob : IJob {

            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public float4x4* Matrices;

            [NativeDisableUnsafePtrRestriction]
            public float3* Center;

            public int Size;

            public void Execute() {
                var center = float3.zero;
                for (int i = 0; i < Size; i++) {
                    float4x4 m = Matrices[i];
                    center += m.Position();
                }

                *Center = center /= Size;
            }
        }

        [BurstCompile]
        public struct CopyMatrixJob : IJobParallelFor {

            [WriteOnly]
            [NativeDisableUnsafePtrRestriction]
            public float4x4* Dst;

            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public float4x4* Src;

            public void Execute(int index) {
                Dst[index] = Src[index];
            }
        }

        [BurstCompile]
        public struct BatchedBoidJob : IJobParallelFor {

            public BoidWeights Weights;
            public float       Time;
            public float       DeltaTime;
            public float       MaxDist;
            public float       Speed;
            public float       RotationSpeed;
            public int         Size;
            public float3      Goal;

            [ReadOnly]
            public NativeArray<float> NoiseOffsets;

            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public float4x4* Src;

            [WriteOnly]
            [NativeDisableUnsafePtrRestriction]
            public float4x4* Dst;

            public void Execute(int index) {
                float4x4 current  = Src[index];
                var currentPos    = current.Position();
                var perceivedSize = Size - 1;

                var separation = float3.zero;
                var alignment  = float3.zero;
                var cohesion   = float3.zero;
                var tendency   = math.normalizesafe(Goal - currentPos) * Weights.TendencyWeight;

                for (int i = 0; i < Size; i++) {
                    if (i == index) {
                        continue;
                    }

                    float4x4 b = Src[i];
                    var other = b.Position();

                    // Perform separation
                    separation += TransformExtensions.SeparationVector(currentPos, other, MaxDist);

                    // Perform alignment
                    alignment  += b.Forward();

                    // Perform cohesion
                    cohesion   += other;
                }

                var avg = 1f / perceivedSize;

                alignment     *= avg;
                cohesion      *= avg;
                cohesion       = math.normalizesafe(cohesion - currentPos);
                var direction  = separation +
                                 Weights.AlignmentWeight * alignment +
                                 cohesion +
                                 Weights.TendencyWeight * tendency;

                var targetRotation = current.Forward().QuaternionBetween(math.normalizesafe(direction));
                var finalRotation  = current.Rotation();

                if (!targetRotation.Equals(current.Rotation())) {
                    finalRotation = math.lerp(finalRotation.value, targetRotation.value, RotationSpeed * DeltaTime);
                }

                var pNoise = math.abs(noise.cnoise(new float2(Time, NoiseOffsets[index])) * 2f - 1f);
                var speedNoise = Speed * (1f + pNoise * Weights.NoiseWeight * 0.9f);
                var finalPosition = currentPos + current.Forward() * speedNoise * DeltaTime;

                Dst[index] = float4x4.TRS(finalPosition, finalRotation, Vector3.one);
            }
        }
    }
}
