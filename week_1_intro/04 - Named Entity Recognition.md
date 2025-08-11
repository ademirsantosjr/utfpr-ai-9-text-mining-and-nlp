# Named Entity Recognition with spaCy

## What are Named Entities?

The spaCy library offers an NLP task for Named Entity Recognition (NER). Named entities are references to people, places, organizations, events, and other specific concepts that can be identified in a text.

## Accessing Named Entities

After processing a text with spaCy, the named entities can be accessed through the `ents` property of the `Document` object. This property is iterable, allowing you to go through all the entities identified in the text.

Each entity has different properties, such as:

*   **ent.text:** The text that identifies the named entity.
*   **ent.label_:** An acronym that represents the type of the entity (e.g., `ORG` for organization, `LOC` for location, `PER` for person).

spaCy uses a coding model for these acronyms. It is possible to consult the documentation to understand the meaning of each one.

## Visualization of Named Entities

The `displacy` library also allows you to render a visualization of the named entities, highlighting them visually in the text. This facilitates the identification and analysis of the found entities.

## Model Limitations

The quantity and accuracy of the named entities that spaCy can identify are directly related to the language model used. Simpler models, such as `pt_core_news_sm`, may have limitations in identifying entities, especially in Portuguese. Larger and more complex models, such as `pt_core_news_lg`, tend to offer better results.