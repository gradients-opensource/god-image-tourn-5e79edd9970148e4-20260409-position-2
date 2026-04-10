from typing import List
from collections import defaultdict
import re

IMAGE_STYLES = [
    "Watercolor Painting",
    "Oil Painting",
    "Digital Art",
    "Pencil Sketch",
    "Comic Book Style",
    "Cyberpunk",
    "Steampunk",
    "Impressionist",
    "Pop Art",
    "Minimalist",
    "Gothic",
    "Art Nouveau",
    "Pixel Art",
    "Anime",
    "3D Render",
    "Low Poly",
    "Photorealistic",
    "Vector Art",
    "Abstract Expressionism",
    "Realism",
    "Futurism",
    "Cubism",
    "Surrealism",
    "Baroque",
    "Renaissance",
    "Fantasy Illustration",
    "Sci-Fi Illustration",
    "Ukiyo-e",
    "Line Art",
    "Black and White Ink Drawing",
    "Graffiti Art",
    "Stencil Art",
    "Flat Design",
    "Isometric Art",
    "Retro 80s Style",
    "Vaporwave",
    "Dreamlike",
    "High Fantasy",
    "Dark Fantasy",
    "Medieval Art",
    "Art Deco",
    "Hyperrealism",
    "Sculpture Art",
    "Caricature",
    "Chibi",
    "Noir Style",
    "Lowbrow Art",
    "Psychedelic Art",
    "Vintage Poster",
    "Manga",
    "Holographic",
    "Kawaii",
    "Monochrome",
    "Geometric Art",
    "Photocollage",
    "Mixed Media",
    "Ink Wash Painting",
    "Charcoal Drawing",
    "Concept Art",
    "Digital Matte Painting",
    "Pointillism",
    "Expressionism",
    "Sumi-e",
    "Retro Futurism",
    "Pixelated Glitch Art",
    "Neon Glow",
    "Street Art",
    "Acrylic Painting",
    "Bauhaus",
    "Flat Cartoon Style",
    "Carved Relief Art",
    "Fantasy Realism",
]

PERSON_KEYWORDS = {
    "man", "woman", "person", "people", "portrait", "face", "faces",
    "boy", "girl", "child", "children", "kid", "baby",
    "he", "she", "him", "her", "his",
    "standing", "sitting", "walking", "running", "posing",
    "wearing", "dressed", "outfit", "clothes",
    "looking", "smiling", "laughing", "holding",
    "headshot", "selfie", "photo of", "photograph of",
    "young", "old", "elderly", "teenager", "adult",
    "male", "female", "gentleman", "lady",
}


def has_person_keywords(prompt: str) -> bool:
    """Check if a prompt contains person-related keywords."""
    prompt_lower = prompt.lower()
    return any(re.search(rf'\b{re.escape(kw)}\b', prompt_lower) for kw in PERSON_KEYWORDS)


def detect_styles_in_prompts(prompts: List[str], style_list=IMAGE_STYLES):
    """
    Analyzes a list of prompts to detect art styles and their appearance percentages.
    
    If majority of prompts contain person keywords, override and return empty
    (classify as person, not style) to avoid false positives like
    "minimalist portrait of a woman" being classified as style.
    
    Args:
        prompts: List of text prompts to analyze
        
    Returns:
        Dictionary mapping style names to their appearance percentages (0-100)
    """
    if not prompts:
        return []
    
    # Person keyword override: kalau mayoritas prompts tentang person, skip style detection
    person_count = sum(1 for p in prompts if has_person_keywords(p))
    person_ratio = person_count / len(prompts) if prompts else 0
    
    if person_ratio > 0.5:
        print(f"Person keywords detected in {person_count}/{len(prompts)} prompts ({person_ratio:.0%}). Classifying as person.", flush=True)
        return []
    
    # Create pattern variations for each style
    style_patterns = {}
    for style in style_list:
        patterns = _create_style_patterns(style)
        style_patterns[style] = patterns
    
    # Count matches for each style
    style_matches = defaultdict(int)
    total_prompts = len(prompts)
    
    for prompt in prompts:
        prompt_lower = prompt.lower()
        matched_styles = set()  # Track styles matched in this prompt to avoid double counting
        
        # Sort styles by length (descending) to prioritize longer/more specific matches
        sorted_styles = sorted(style_list, key=len, reverse=True)
        
        for style in sorted_styles:
            patterns = style_patterns[style]
            
            # Check if any pattern matches
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    matched_styles.add(style)
                    break
        
        # Filter out styles that are substrings of other matched styles
        filtered_matches = _filter_substring_styles(matched_styles)
        
        # Count matches
        for style in filtered_matches:
            style_matches[style] += 1
    
    # Calculate percentages
    style_percentages = {}
    for style, count in style_matches.items():
        percentage = (count / total_prompts) * 100
        style_percentages[style] = round(percentage, 2)
    
    return [(style, percentage) for style, percentage in style_percentages.items() if percentage >= 25]

def _create_style_patterns(style: str) -> List[str]:
    """
    Creates regex patterns for detecting style variations in text.
    
    Args:
        style: The style name to create patterns for
        
    Returns:
        List of regex patterns
    """
    patterns = []
    style_lower = style.lower()
    
    # Exact match with word boundaries
    escaped_style = re.escape(style_lower)
    patterns.append(rf'\b{escaped_style}\b')
    
    # Create adjective form patterns
    adjective_patterns = _create_adjective_patterns(style_lower)
    patterns.extend(adjective_patterns)
    
    # Handle compound styles - but be more restrictive
    # Only create base patterns for styles that are commonly abbreviated
    compound_exceptions = {
        'watercolor painting': ['watercolor'],
        'oil painting': ['oil paint', 'oil painted'],
        'acrylic painting': ['acrylic paint', 'acrylic painted'],
        'pencil sketch': ['pencil drawing', 'pencil sketched', 'pencil'],
        'digital art': ['digital artwork', 'digitally created'],
        'pixel art': ['pixel graphics', '8-bit art', '8 bit art'],  # Removed 'pixelated' to avoid conflict
        'vector art': ['vector graphics', 'vector illustration'],
        'line art': ['line drawing', 'line work'],
        '3d render': ['3d rendered', '3d rendering', 'three dimensional render'],
        'comic book style': ['comic book', 'comic style'],
        'black and white ink drawing': ['black and white ink', 'ink drawing'],
        'sci-fi illustration': ['sci-fi', 'science fiction'],
        'retro 80s style': ['80s style', 'eighties style'],
        'flat design': ['flat style'],
        'low poly': ['low polygon'],
    }
    
    if style_lower in compound_exceptions:
        for variant in compound_exceptions[style_lower]:
            escaped_variant = re.escape(variant)
            patterns.append(rf'\b{escaped_variant}\b')
    
    return patterns

def _create_adjective_patterns(style: str) -> List[str]:
    """
    Creates patterns for adjective forms of style names.
    """
    patterns = []
    
    # Handle specific style mappings directly - only for styles that commonly appear as adjectives
    style_mappings = {
        'expressionism': ['expressionist', 'expressionistic'],
        'impressionist': ['impressionist', 'impressionistic'], 
        'cubism': ['cubist', 'cubistic'],
        'surrealism': ['surrealist', 'surrealistic'],
        'futurism': ['futurist', 'futuristic'],
        'realism': ['realist'],
        'hyperrealism': ['hyperrealistic', 'hyperrealist'],
        'minimalist': ['minimalist', 'minimalistic'],
        'abstract expressionism': ['abstract expressionist', 'abstract expressionistic'],
        'photorealistic': ['photorealistic', 'photo realistic'],
        'cyberpunk': ['cyberpunk style', 'cyberpunk aesthetic'],
        'steampunk': ['steampunk style', 'steampunk aesthetic'],
        'gothic': ['gothic style', 'gothic aesthetic'],
        'baroque': ['baroque style'],
        'renaissance': ['renaissance style'],
        'art nouveau': ['art nouveau style'],
        'art deco': ['art deco style'],
        'pop art': ['pop art style'],
        'anime': ['anime style', 'anime-style'],
        'manga': ['manga style', 'manga-style'],
        'chibi': ['chibi style'],
        'kawaii': ['kawaii style'],
        'noir style': ['noir', 'film noir'],
        'retro futurism': ['retro futuristic', 'retrofuturistic'],
        'vaporwave': ['vaporwave aesthetic', 'vaporwave style'],
        'dreamlike': ['dreamlike', 'dream-like'],
        'psychedelic art': ['psychedelic', 'psychedelic style'],
        'monochrome': ['monochromatic'],
    }
    
    # Check for direct mappings
    if style in style_mappings:
        for variant in style_mappings[style]:
            patterns.append(rf'\b{re.escape(variant)}\b')
    
    # Apply limited general transformations - only for "-ism" endings
    if style.endswith('ism') and style not in style_mappings:
        base = style[:-3]  # Remove 'ism'
        patterns.append(rf'\b{re.escape(base + "ist")}\b')
        patterns.append(rf'\b{re.escape(base + "istic")}\b')
    
    return patterns

def _filter_substring_styles(matched_styles: set) -> set:
    """
    Filters out styles that are substrings of other matched styles.
    Keeps only the longest/most specific styles.
    
    Args:
        matched_styles: Set of matched style names
        
    Returns:
        Filtered set with substring styles removed
    """
    if len(matched_styles) <= 1:
        return matched_styles
    
    # Sort by length (descending) to check longer styles first
    sorted_matches = sorted(matched_styles, key=len, reverse=True)
    filtered_styles = set()
    
    for current_style in sorted_matches:
        current_lower = current_style.lower()
        is_substring = False
        
        # Check if current style is a substring of any already filtered style
        for existing_style in filtered_styles:
            existing_lower = existing_style.lower()
            if current_lower in existing_lower and current_style != existing_style:
                is_substring = True
                break
        
        # Only add if it's not a substring of an existing style
        if not is_substring:
            filtered_styles.add(current_style)
    
    return filtered_styles