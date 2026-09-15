#!/bin/bash

ROOT="$HOME/BackgroundMellow_ismir/backgroundMellow/backend"
OUTPUT="$ROOT/directory_structure.txt"

MAX_FILES=20

print_tree() {
    local dir="$1"
    local prefix="$2"

    # --------------------------------------------------
    # Directories
    # --------------------------------------------------

    while IFS= read -r -d '' path; do
        name=$(basename "$path")

        # Ignore these directories completely
        if [[ "$name" == "node_modules" || "$name" == "__pycache__" ]]; then
            continue
        fi

        # tango_new: show it but don't enter it
        if [[ "$name" == "tango_new" ]]; then
            echo "${prefix}├── ${name}/ [not expanded]"
            continue
        fi

        echo "${prefix}├── ${name}/"

        print_tree "$path" "${prefix}│   "

    done < <(
        find "$dir" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -print0 | sort -z
    )


    # --------------------------------------------------
    # Files
    # --------------------------------------------------

    files=()

    while IFS= read -r -d '' path; do
        name=$(basename "$path")

        # Ignore unwanted files
        [[ "$name" == "directory_structure.txt" ]] && continue
        [[ "$name" == *.md ]] && continue
        [[ "$name" == *.yml ]] && continue
        [[ "$name" == *.yaml ]] && continue

        files+=("$name")

    done < <(
        find "$dir" \
            -mindepth 1 \
            -maxdepth 1 \
            -type f \
            -print0 | sort -z
    )


    # --------------------------------------------------
    # Print maximum 20 files
    # --------------------------------------------------

    total=${#files[@]}

    if (( total > MAX_FILES )); then
        limit=$MAX_FILES
    else
        limit=$total
    fi

    for ((i=0; i<limit; i++)); do
        echo "${prefix}├── ${files[$i]}"
    done


    # --------------------------------------------------
    # Mention omitted files
    # --------------------------------------------------

    if (( total > MAX_FILES )); then
        omitted=$((total - MAX_FILES))

        echo "${prefix}└── [Only 20 files written; ${omitted} more files omitted]"
    fi
}


# ------------------------------------------------------
# Generate output
# ------------------------------------------------------

{
    echo "Directory structure for:"
    echo "$ROOT"
    echo
    echo "$(basename "$ROOT")/"

    print_tree "$ROOT" "    "

} > "$OUTPUT"


echo "Structure saved to:"
echo "$OUTPUT"