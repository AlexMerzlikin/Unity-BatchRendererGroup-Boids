using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace ThousandAnt.Boids
{
    public unsafe class BoidsPointerOnlyCopyToFloat4
    {
        [BurstCompile]
        public struct AverageCenterJob : IJob
        {
            [ReadOnly] [NativeDisableUnsafePtrRestriction]
            public float4x4* Matrices;

            [NativeDisableUnsafePtrRestriction] public float3* Center;

            public int Size;

            public void Execute()
            {
                var center = float3.zero;
                for (var i = 0; i < Size; i++)
                {
                    var m = Matrices[i];
                    center += m.Position();
                }

                *Center = center / Size;
            }
        }

        [BurstCompile]
        public struct CopyMatrixJob : IJobParallelFor
        {
            [WriteOnly] [NativeDisableUnsafePtrRestriction]
            public float4x4* Dst;

            [ReadOnly] [NativeDisableUnsafePtrRestriction]
            public float4x4* Src;

            public void Execute(int index)
            {
                Dst[index] = Src[index];
            }
        }

        [BurstCompile]
        public struct BatchedBoidsJob : IJobParallelFor
        {
            public BoidWeights Weights;
            public float Time;
            public float DeltaTime;
            public float MaxDist;
            public float Speed;
            public float RotationSpeed;
            public int Size;
            public float3 Goal;

            [ReadOnly] public NativeArray<float> NoiseOffsets;

            [ReadOnly] [NativeDisableUnsafePtrRestriction]
            public float4x4* Src;

            [WriteOnly] [NativeDisableUnsafePtrRestriction]
            public float4x4* Dst;

            [WriteOnly] [NativeDisableParallelForRestriction]
            public NativeArray<Vector4> DataBuffer;

            private const int PositionOffset = 4;

            public void Execute(int index)
            {
                var current = Src[index];
                var currentPos = current.Position();
                var perceivedSize = Size - 1;

                var separation = float3.zero;
                var alignment = float3.zero;
                var cohesion = float3.zero;
                var tendency = math.normalizesafe(Goal - currentPos) * Weights.TendencyWeight;

                for (var i = 0; i < Size; i++)
                {
                    if (i == index)
                    {
                        continue;
                    }

                    var b = Src[i];
                    var other = b.Position();

                    // Perform separation
                    separation += TransformExtensions.SeparationVector(currentPos, other, MaxDist);

                    // Perform alignment
                    alignment += b.Forward();

                    // Perform cohesion
                    cohesion += other;
                }

                var avg = 1f / perceivedSize;

                alignment *= avg;
                cohesion *= avg;
                cohesion = math.normalizesafe(cohesion - currentPos);
                var direction = separation +
                                Weights.AlignmentWeight * alignment +
                                cohesion +
                                Weights.TendencyWeight * tendency;

                var targetRotation = current.Forward().QuaternionBetween(math.normalizesafe(direction));
                var finalRotation = current.Rotation();

                if (!targetRotation.Equals(current.Rotation()))
                {
                    finalRotation = math.lerp(finalRotation.value, targetRotation.value, RotationSpeed * DeltaTime);
                }

                var pNoise = math.abs(noise.cnoise(new float2(Time, NoiseOffsets[index])) * 2f - 1f);
                var speedNoise = Speed * (1f + pNoise * Weights.NoiseWeight * 0.9f);
                var finalPosition = currentPos + current.Forward() * speedNoise * DeltaTime;

                Dst[index] = float4x4.TRS(finalPosition, finalRotation, Vector3.one);

                DataBuffer[PositionOffset + index * 3 + 0] = new Vector4(Src[index].c0.x, Src[index].c0.y, Src[index].c0.z, Src[index].c1.x);
                DataBuffer[PositionOffset + index * 3 + 1] = new Vector4(Src[index].c1.y, Src[index].c1.z, Src[index].c2.x, Src[index].c2.y);
                DataBuffer[PositionOffset + index * 3 + 2] = new Vector4(Src[index].c2.z, Src[index].c3.x, Src[index].c3.y, Src[index].c3.z);

                var offset = Size * 3;
                var inverse = Matrix4x4.Inverse(Src[index]);
                DataBuffer[PositionOffset + index * 3 + 0 + offset] = new Vector4(inverse.m00, inverse.m10, inverse.m20, inverse.m01);
                DataBuffer[PositionOffset + index * 3 + 1 + offset] = new Vector4(inverse.m11, inverse.m21, inverse.m02, inverse.m12);
                DataBuffer[PositionOffset + index * 3 + 2 + offset] = new Vector4(inverse.m22, inverse.m03, inverse.m13, inverse.m23);
            }
        }
    }
}