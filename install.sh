cd ~/github/D-VAQEM
while IFS= read -r pkg; do
    [[ -z "$pkg" || "$pkg" == \#* ]] && continue
    echo ">>> Installing: $pkg"
    pip install "$pkg" || echo "!!! Skipped: $pkg"
done < requirments.txt
