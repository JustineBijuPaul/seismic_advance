from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TutorialStep:
    title: str
    content: str
    image_url: str = None
    video_url: str = None

@dataclass
class SeismicZone:
    name: str
    risk_level: str
    description: str
    characteristics: List[str]
    precautions: List[str]

class EducationalContent:
    def __init__(self):
        self.tutorials = {
            'before': [
                TutorialStep(
                    title="Create an Emergency Plan",
                    content="Learn how to develop a comprehensive family emergency plan.",
                    image_url="static/education/emergency_plan.jpg"
                ),
                TutorialStep(
                    title="Prepare Emergency Kit",
                    content="Essential items to include in your earthquake preparedness kit.",
                    image_url="static/education/emergency_kit.jpg"
                ),
                TutorialStep(
                    title="Secure Your Home",
                    content="Steps to earthquake-proof your living space.",
                    image_url="static/education/home_safety.jpg"
                )
            ],
            'during': [
                TutorialStep(
                    title="Drop, Cover, and Hold On",
                    content="The correct way to protect yourself during an earthquake.",
                    video_url="static/education/drop_cover_hold.mp4"
                ),
                TutorialStep(
                    title="Indoor Safety",
                    content="What to do if you're indoors during an earthquake.",
                    image_url="static/education/indoor_safety.jpg"
                ),
                TutorialStep(
                    title="Outdoor Safety",
                    content="How to stay safe if you're outdoors during an earthquake.",
                    image_url="static/education/outdoor_safety.jpg"
                )
            ],
            'after': [
                TutorialStep(
                    title="Check for Injuries",
                    content="How to assess and provide basic first aid.",
                    image_url="static/education/first_aid.jpg"
                ),
                TutorialStep(
                    title="Evaluate Building Safety",
                    content="Signs of structural damage to look for.",
                    image_url="static/education/building_damage.jpg"
                ),
                TutorialStep(
                    title="Recovery Steps",
                    content="Important steps for post-earthquake recovery.",
                    image_url="static/education/recovery.jpg"
                )
            ]
        }

        self.seismic_zones = [
            SeismicZone(
                name="High Risk Zone",
                risk_level="Severe",
                description="Areas near active fault lines with frequent seismic activity.",
                characteristics=[
                    "Located near tectonic plate boundaries",
                    "History of major earthquakes",
                    "Visible fault lines",
                    "Regular seismic activity"
                ],
                precautions=[
                    "Regular building inspections",
                    "Strict building codes",
                    "Emergency drills",
                    "Early warning systems"
                ]
            ),
            SeismicZone(
                name="Moderate Risk Zone",
                risk_level="Moderate",
                description="Regions with occasional seismic activity.",
                characteristics=[
                    "Secondary fault lines",
                    "Infrequent earthquakes",
                    "Moderate ground stability"
                ],
                precautions=[
                    "Building reinforcement",
                    "Emergency preparedness",
                    "Risk assessment"
                ]
            ),
            SeismicZone(
                name="Low Risk Zone",
                risk_level="Low",
                description="Areas with rare seismic activity.",
                characteristics=[
                    "Stable geological formation",
                    "No major fault lines",
                    "Rare seismic events"
                ],
                precautions=[
                    "Basic preparedness",
                    "General awareness",
                    "Building safety"
                ]
            )
        ]

        self.safety_guidelines = {
            "home": [
                "Secure heavy furniture to walls",
                "Install latches on cabinets",
                "Keep emergency supplies accessible",
                "Know utility shutoff locations"
            ],
            "workplace": [
                "Know evacuation routes",
                "Identify safe spots",
                "Keep emergency contact list",
                "Regular safety drills"
            ],
            "public": [
                "Stay away from buildings",
                "Move to open areas",
                "Avoid bridges and overpasses",
                "Follow emergency instructions"
            ]
        }

    def get_tutorial(self, phase: str) -> List[TutorialStep]:
        return self.tutorials.get(phase, [])

    def get_seismic_zone_info(self, risk_level: str = None) -> List[SeismicZone]:
        if risk_level:
            return [zone for zone in self.seismic_zones if zone.risk_level.lower() == risk_level.lower()]
        return self.seismic_zones

    def get_safety_guidelines(self, location: str = None) -> Dict[str, List[str]]:
        if location:
            return {location: self.safety_guidelines.get(location, [])}
        return self.safety_guidelines
