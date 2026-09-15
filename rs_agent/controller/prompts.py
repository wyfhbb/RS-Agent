"""Prompt templates for RS-Agent."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Keep the structured-chat contract local so running the agent does not require
# downloading and deserializing an unversioned public Hub prompt.
STRUCTURED_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Respond to the human as helpfully and accurately as possible.
You have access to the following tools:

{tools}

Return exactly one JSON action at a time, inside a Markdown JSON code block.
Valid "action" values are "Final Answer" or one of: {tool_names}.
Use "action_input" for the tool input or the final response to the human.

```json
{{"action": "tool_name", "action_input": "tool input"}}
```

Wait for the actual tool observation before choosing the next action.
Follow the user's requested operation order. Do not invent tool observations
or output file paths. When finished, return:

```json
{{"action": "Final Answer", "action_input": "Final response to the human"}}
```""",
        ),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}\n\n{agent_scratchpad}\n\nReturn one JSON action."),
    ]
)

TASK_TYPES = [
    "Super_Resolution",
    "Denoising",
    "Captioning",
    "Optical_Detection",
    "Optical_Plane_Type",
    "Scene_Classification",
    "SAR_Detection",
    "SAR_Plane_Type",
    "Knowledge_Search",
    "Building_Damage_Detection",
    "Building_Extraction",
    "Road_Extraction",
    "Horizontal_Object_Detection",
    "Rotated_Object_Detection",
    "Semantic_Segmentation",
    "Land_Use_Classification",
    "Image_Dehazing",
    "Cloud_Removal",
]

TASK_TYPE_PROMPT = """
Given the question: "{question}", identify the most relevant task type from the following list
(Return the result as an array):
- Super_Resolution: Tasks that involve improving the resolution of the image.
- Denoising: Tasks that involve removing noise or unwanted distortions from the image.
- Captioning: Tasks that involve generating textual descriptions or summaries of the image.
- Optical_Detection: Tasks that involve detecting and counting specific targets in optical images.
- Optical_Plane_Type: Tasks that involve determining the type of plane in an optical image.
- Scene_Classification: Tasks that involve identifying the type of scene shown in the image.
- SAR_Detection: Tasks that involve detecting and counting specific targets in SAR images.
- SAR_Plane_Type: Tasks that involve determining the type of plane in a SAR image.
- Knowledge_Search: Tasks that involve searching for information about a specific plane or aircraft.
- Building_Damage_Detection: Tasks that involve assessing building damage after a disaster in an image.
- Building_Extraction: Tasks that involve extracting buildings from an image.
- Road_Extraction: Tasks that involve extracting roads from an image.
- Horizontal_Object_Detection: Tasks that involve detecting objects with horizontally aligned bounding boxes.
- Rotated_Object_Detection: Tasks that involve detecting objects with rotated bounding boxes in the image.
- Semantic_Segmentation: Tasks that involve pixel-wise classification or segmentation of objects or regions.
- Land_Use_Classification: Tasks that involve classifying land use or land cover in the image.
- Image_Dehazing: Tasks that involve removing haze or fog from an image.
- Cloud_Removal: Tasks that involve removing clouds from an image.

Return the result as an array. Please return only one task type. Do not provide any explanations or additional text.
"""

TASK_TYPES_RSCHATGPT = [
    "Object_Counting",
    "Object_Detection",
    "Landuse_Segmentation",
    "Instance_Segmentation",
    "Edge_Detection",
    "Image_Caption",
    "Scene_Classification",
]

TASK_TYPE_PROMPT_RSCHATGPT = """
Given the question: "{question}", identify the most relevant task type from the following list
(Return the result as an array):
- Object_Counting: Tasks that involve counting the number of specific objects in the image.
- Object_Detection: Tasks that need to detect the bounding box of the certain objects in the picture.
- Landuse_Segmentation: Tasks that involve segmenting different land use types in the image.
- Instance_Segmentation: Tasks that need to apply man-made instance segmentation for the image.
- Edge_Detection: Tasks that involve detecting the boundaries or edges of objects in the image.
- Image_Caption: Tasks that involve summarizing or describing the image.
- Scene_Classification: Tasks that need to determine which specific scene the image belongs to.

Return the result as an array. Please return only one task type. Do not provide any explanations or additional text.
"""
