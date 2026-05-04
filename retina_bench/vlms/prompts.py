def get_zero_shot_prompts(task: str) -> dict:
    # Generates positive and negative text prompts for Zero-Shot
    templates = {
        "binary_dr": {
            0: "A fundus photograph of a healthy retina without diabetic retinopathy.",
            1: "A fundus photograph of a retina with diabetic retinopathy."
        },
        "referable_dr": {
            0: "A fundus photograph showing no referable diabetic retinopathy.",
            1: "A fundus photograph showing referable diabetic retinopathy."
        },
        "glaucoma": {
            0: "A fundus photograph of a healthy eye without glaucoma.",
            1: "A fundus photograph showing signs of glaucoma."
        }
    }
    return templates.get(task, {
        0: f"A photo without {task}",
        1: f"A photo with {task}"
    })
