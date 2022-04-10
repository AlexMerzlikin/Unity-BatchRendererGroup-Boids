Shader "CustomURP/Unlit Fish"
{
    Properties
    {
        _BaseMap ("Base Texture", 2D) = "white" {}
        _BaseColor ("Base Colour", Color) = (1, 1, 1, 1)
        _EffectRadius("Wave Effect Radius",Range(0.0,1.0)) = 0.5
        _WaveSpeed("Wave Speed", Range(0.0,100.0)) = 3.0
        _WaveHeight("Wave Height", Range(0.0,30.0)) = 5.0
        _WaveDensity("Wave Density", Range(0.0001,1.0)) = 0.007
        _Yoffset("Y Offset",Float) = 0.0
        _Threshold("Threshold",Range(0,30)) = 3
        _HeadLimit("HeadLimit", Range(-5, 5)) = 0.05
    }

    SubShader
    {
        Tags
        {
            "RenderPipeline"="UniversalPipeline" "Queue"="Geometry"
        }

        Pass
        {
            Name "Forward"
            Tags
            {
                "LightMode"="UniversalForward"
            }

            Cull Back

            HLSLPROGRAM
            #pragma exclude_renderers gles gles3 glcore
            #pragma target 4.5
            #pragma vertex UnlitPassVertex
            #pragma fragment UnlitPassFragment
            #pragma multi_compile_instancing
            #pragma instancing_options renderinglayer
            #pragma multi_compile _ DOTS_INSTANCING_ON
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/UnityInstancing.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
                float4 color : COLOR;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                float4 color : COLOR;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            CBUFFER_START(UnityPerMaterial)
            float4 _BaseMap_ST;
            float4 _BaseColor;
            half _EffectRadius;
            half _WaveSpeed;
            half _WaveHeight;
            half _WaveDensity;
            half _Yoffset;
            int _Threshold;
            half _HeadLimit;
            CBUFFER_END

            #ifdef UNITY_DOTS_INSTANCING_ENABLED
                UNITY_DOTS_INSTANCING_START(MaterialPropertyMetadata)
                    UNITY_DOTS_INSTANCED_PROP(float4, _BaseColor)
                    UNITY_DOTS_INSTANCED_PROP(half, _EffectRadius)
                    UNITY_DOTS_INSTANCED_PROP(half, _WaveSpeed)
                    UNITY_DOTS_INSTANCED_PROP(half, _WaveHeight)
                    UNITY_DOTS_INSTANCED_PROP(half, _WaveDensity)
                    UNITY_DOTS_INSTANCED_PROP(half, _Yoffset)
                    UNITY_DOTS_INSTANCED_PROP(half, _Threshold)
                    UNITY_DOTS_INSTANCED_PROP(half, _HeadLimit)
                UNITY_DOTS_INSTANCING_END(MaterialPropertyMetadata)
                #define _BaseColor UNITY_ACCESS_DOTS_INSTANCED_PROP_WITH_DEFAULT(float4, _BaseColor)
                #define _EffectRadius UNITY_ACCESS_DOTS_INSTANCED_PROP_WITH_DEFAULT(half, _EffectRadius)
                #define _WaveSpeed UNITY_ACCESS_DOTS_INSTANCED_PROP_WITH_DEFAULT(half, _WaveSpeed)
                #define _WaveHeight UNITY_ACCESS_DOTS_INSTANCED_PROP_WITH_DEFAULT(half, _WaveHeight)
                #define _WaveDensity UNITY_ACCESS_DOTS_INSTANCED_PROP_WITH_DEFAULT(half, _WaveDensity)
                #define _Yoffset UNITY_ACCESS_DOTS_INSTANCED_PROP_WITH_DEFAULT(half, _Yoffset)
                #define _Threshold UNITY_ACCESS_DOTS_INSTANCED_PROP_WITH_DEFAULT(half, _Threshold)
                #define _HeadLimit UNITY_ACCESS_DOTS_INSTANCED_PROP_WITH_DEFAULT(half, _HeadLimit)
            #endif
            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);

            Varyings UnlitPassVertex(Attributes input)
            {
                Varyings output;

                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);

                const half z = input.positionOS.z;
                half sinUse;
                if (z > _HeadLimit)
                {
                    sinUse = sin(-_Time.y * _WaveSpeed + z * _WaveDensity * _HeadLimit);
                }
                else
                {
                    sinUse = sin(-_Time.y * _WaveSpeed + z * _WaveDensity * z);
                }

                half yValue = input.positionOS.y - _Yoffset;
                half yDirScaling = clamp(pow(yValue * _EffectRadius, _Threshold), 0.0, 1.0);
                input.positionOS.x = input.positionOS.x + sinUse * _WaveHeight * yDirScaling;

                const VertexPositionInputs positionInputs = GetVertexPositionInputs(input.positionOS.xyz);
                output.positionCS = positionInputs.positionCS;
                output.uv = TRANSFORM_TEX(input.uv, _BaseMap);
                output.color = input.color;
                return output;
            }

            half4 UnlitPassFragment(Varyings input) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(input);
                const half4 baseMap = half4(SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, input.uv));
                return baseMap * _BaseColor * input.color;
            }
            ENDHLSL
        }
    }
}