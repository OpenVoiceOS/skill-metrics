"""this quick and dirty script gathers intent data  from several datasets
normalization of intent labels is applied to ensure the labels correspond to a OVOS bus message that triggers an intent
This means any trained classifier should work out of the box with OVOS
normalization is a bit hacky right now, some of the data sources have old/polluted data, in the future these should probably be unified and this script simplified

expectations:
- OpenVoiceOS/lang-support-tracker will always provide the latest data and is always valid, any strings removed should be dropped and assumed to have been removed because they were invalid
- the llm augmented dataset might receive new entries over time, but we probably should not rely too much on it, mostly meant to cover unbalanced intents that need more training data
- a new test set should be created and validated by humans, this script is meant to create the latest **train** set
- eventually model training can be automated (github actions?) to automatically handle new OVOS default skills (via OpenVoiceOS/lang-support-tracker)
"""
import json
import pandas as pd

# Dataset sources
csv_sources = [
    "https://huggingface.co/datasets/Jarbas/ovos_intent_examples/resolve/main/dataset.csv", # TODO - merge into llm test set
    "https://huggingface.co/datasets/Jarbas/music_queries_templates/resolve/main/music_templates.csv",
    "https://huggingface.co/datasets/Jarbas/ovos-common-query-intents/resolve/main/common_query.csv",
    "https://huggingface.co/datasets/Jarbas/ovos-llm-augmented-intents/resolve/main/augmented.csv",
    "https://huggingface.co/datasets/Jarbas/ovos-weather-intents/resolve/main/weather_intents_en.csv",  # TODO - merge into llm test set
    "https://huggingface.co/datasets/Jarbas/core_intents/resolve/main/dataset.csv"  # TODO - merge into llm test set
    # FUTURE NOTE: "shutdown" intent (as in, power off) will get confused with "stop", but "system skill" does not yet exist
]

langs = ["en", "pt", "eu", "es", "gl", "nl", "fr", "de", "ca", "it", "da"]
github_sources = [
    f"https://raw.githubusercontent.com/OpenVoiceOS/lang-support-tracker/refs/heads/dev/skills/intents_{lang}.csv"
    for lang in langs
]
csv_sources += github_sources

BLACKLIST_SKILLS = [
    "ovos-skill-local-media.openvoiceos",
    "ovos-skill-spotify.openvoiceos",
    # "ovos-skill-confucius-quotes.openvoiceos",
    # "ovos-skill-color-picker.krisgesling.openvoiceos",
    # "ovos-skill-fuster-quotes.openvoiceos"
]
SKILL_REPLACEMENTS = {

}

BLACKLIST_INTENTS = [
    "no_secondary_langs.intent",  # dialog not intent

]
BLACKLIST_LABELS = [

]

LABEL_FIXES = {
    "ovos-skill-ddg.openvoiceos:search_wolfie.intent": "ovos-skill-wolfie.openvoiceos:search_wolfie.intent",
    "ovos-skill-ddg.openvoiceos:common_query": "common_query:common_query",
    "ovos-skill-confucius-quotes.openvoiceos:common_query": "ovos-skill-confucius-quotes.openvoiceos:who.intent",
    "ovos-skill-fuster-quotes.openvoiceos:common_query": "ovos-skill-fuster-quotes.openvoiceos:who.intent",
    "ovos-skill-volume.openvoiceos:volume.mute.intent.toggle.intent": "ovos-skill-volume.openvoiceos:volume.mute.toggle.intent"
}

INTENT_REPLACEMENTS = {  # merge similar intents
    "what.date.is.it.intent": "current_date.intent",
    "handle_day_for_date": "weekday.for.date.intent",
    "handle_query_relative_date": "time.until.intent",
    "howto.intent": "wikihow.intent",
    "HowAreYou.intent": "Greetings.intent",
    "handle_show_time": "what.time.is.it.intent",
    "handle_query_time": "what.time.is.it.intent",
    "current_wind.intent": "is_wind.intent",
    "do-i-need-an-umbrella.intent": "is_rain.intent",
    "do.i.need.an.umbrella.intent": "is_rain.intent",
    "volume.mute": "volume.mute.intent",
    "handle_weekday": "what.weekday.is.it.intent",
    "handle_query_date_simple": "current_date.intent",
    # internal ddg intents
    "age_at_death.intent": "common_query",
    "alma_mater.intent": "common_query",
    "birthdate.intent": "common_query",
    "born.intent": "common_query",
    "children.intent": "common_query",
    "died.intent": "common_query",
    "education.intent": "common_query",
    "fields.intent": "common_query",
    "known_for.intent": "common_query",
    "official_website.intent": "common_query",
    "resting_place.intent": "common_query",
    "thesis.intent": "common_query",
    "who.intent": "common_query"
}


def normalize(text):
    return str(text).lower().replace(",", "").split("/")[-1].replace("  ", " ").strip().strip('"').strip("'").strip("`")


def normalize_label(text):
    n = str(text).strip().strip('"').strip("'").strip("`")
    for k, v in LABEL_FIXES.items():
        n = n.replace(k, v)
    return n


def normalize_domain(text):
    n = str(text).strip().strip('"').strip("'").strip("`")
    n = n.replace(".OpenVoiceOS.openvoiceos", ".openvoiceos")
    for k, v in SKILL_REPLACEMENTS.items():
        n = n.replace(k, v)
    # mistakes noticed in the datasets
    return n.replace("skill-ovos-", "ovos-skill-").lower()


def normalize_intent(text):
    n = str(text).strip().strip('"').strip("'").strip("`")
    for k, v in INTENT_REPLACEMENTS.items():
        n = n.replace(k, v)
    return n.replace(".intent.intent", ".intent")


def load_and_format_csv(url):
    try:
        df = pd.read_csv(url)

        if "github" in url:
            df["lang"] = url.split("_")[-1].split(".csv")[0]

        if "lang" not in df.columns:
            df["lang"] = "en"

        if "music_templates" in url:
            df["domain"] = "ocp"
            df["intent"] = "play"
            df["lang"] = "en"
            df = df.rename(columns={"template": "sentence"})
        elif "weather" in url:
            df["lang"] = "en"
            df["domain"] = "ovos-skill-weather.openvoiceos"
            df["intent"] = df["intent"] + ".intent"  # dataset should end with .intent
            df = df.rename(columns={"example": "sentence"})
        elif "core_intents" in url:
            df["lang"] = "en"
            df["domain"] = "stop"
            df["intent"] = "stop"
            df = df[df["label"] == "stop"]

        if "utterance" in df.columns:
            df = df.rename(columns={"utterance": "sentence"})

        df = df[["lang", "domain", "intent", "sentence"]]

        # Normalize all columns
        df["domain"] = df["domain"].apply(normalize_domain)
        df["intent"] = df["intent"].apply(normalize_intent)
        df["sentence"] = df["sentence"].apply(normalize)

        df["label"] = df["domain"] + ":" + df["intent"]
        df["label"] = df["label"].apply(normalize_label)

        # print(url, df)
        df = df[~df["label"].isin(BLACKLIST_LABELS)]
        # Drop blacklisted domains
        df = df[~df["domain"].isin(BLACKLIST_SKILLS)]
        df = df[~df["intent"].isin(BLACKLIST_INTENTS)]

        #df["lang"] = lang
        return df[["lang", "label", "sentence"]]
    except Exception as e:
        print(f"Failed to load {url}: {e}")
        return pd.DataFrame(columns=["lang", "label", "sentence"])


# Load and merge all datasets
frames = [load_and_format_csv(url) for url in csv_sources]
merged_df = pd.concat(frames, ignore_index=True)

# Deduplicate
merged_df.drop_duplicates(inplace=True)

# Save to CSV
merged_df.to_csv("merged_intents_dataset.csv", index=False)

print(f"Merged dataset created with {len(merged_df)} unique entries")
print(f"Total unique intents: {merged_df['label'].nunique()}")
total_unique_skills = merged_df['label'].apply(lambda x: x.split(":")[0]).nunique()
print(f"Total unique skills (domains): {total_unique_skills}")
sorted_labels = sorted(merged_df['label'].unique())
with open("labels.txt", "w") as f:
    f.write("\n".join(sorted_labels))
print(f"{len(sorted_labels)} labels written to labels.txt")

adapt_labels = [l for l in sorted_labels if not l.endswith(".intent") and "ovos-skill-" in l]
with open("adapt_labels.txt", "w") as f:
    f.write("\n".join(adapt_labels))
print(f"{len(adapt_labels)} labels written to adapt_labels.txt")


padatious_labels = [l for l in sorted_labels if l.endswith(".intent") and "ovos-skill-" in l]
with open("padatious_labels.txt", "w") as f:
    f.write("\n".join(padatious_labels))
print(f"{len(padatious_labels)} labels written to padatious_labels.txt")

metrics = {
    "n_skills": total_unique_skills,
    "n_intents": len(sorted_labels),
    "n_adapt_intents": len(adapt_labels),
    "n_padatious_intents": len(padatious_labels)
}
with open("skill_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

# auto generate a nice markdown report for README.md
with open("README.md", "w") as f:
    f.write("# OVOS Intent Dataset\n\n")
    f.write(f"- Total unique utterances: **{len(merged_df)}**\n")
    f.write(f"- Total unique intents: **{metrics['n_intents']}**\n")
    f.write(f"- Total unique skills (domains): **{metrics['n_skills']}**\n")
    f.write(f"- Adapt-style intents: **{metrics['n_adapt_intents']}**\n")
    f.write(f"- Padatious-style intents: **{metrics['n_padatious_intents']}**\n\n")
    f.write("### Number of Sentences per Intent\n\n")
    f.write("| Intent | Number of Sentences |\n")
    f.write("|--------|----------------------|\n")
    intent_counts = merged_df["label"].value_counts()
    for label, count in intent_counts.items():
        f.write(f"| {label} | {count} |\n")
