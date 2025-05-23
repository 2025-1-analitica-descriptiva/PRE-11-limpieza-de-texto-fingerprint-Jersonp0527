import nltk
import pandas as pd


def load_data(input_file):
    df = pd.read_csv(input_file)

    return df


def create_normalized_key(df):
    """Crea una nueva columna en el df que contenga el key de la columna 'raw_text'"""
    df = df.copy()
    df["key"] = df["raw_text"]

    # Quitar espacios en blanco
    df["key"] = df["key"].str.strip()

    # Convertir a minusculas
    df["key"] = df["key"].str.lower()

    # Elimina el guion entre palabras
    df["key"] = df["key"].str.replace("-", "")

    # Eliminar cualquier simbolo
    df["key"] = df["key"].str.translate(
        str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~¿¡")
    )

    # convertir la llave a lista de palabras
    df["key"] = df["key"].str.split()

    # Tranformar cada palablar
    stremmer = nltk.PorterStemmer()
    df["key"] = df["key"].apply(
        lambda x: [stremmer.stem(word) for word in x]
    )  # lista de raices de palabras

    # Ordene las listas de tokens y remueve duplicados
    df["key"] = df["key"].apply(lambda x: sorted(set(x)))

    # Unir de nuevo las listas de las claves generadas
    df["key"] = df["key"].str.join(" ")

    return df


def generate_cleaned_text(df):
    """Crea la columna 'cleaned_text' en el df"""
    keys = df.copy()

    # Ordene el df por "key" y "text_raw"
    keys = keys.sort_values(by=["key", "raw_text"], ascending=[True, True])

    #
    keys = df.drop_duplicates(subset="key", keep="first")

    key_dict = dict(zip(keys["key"], keys["raw_text"]))

    df["cleaned_text"] = df["key"].map(key_dict)

    return df


def main(input_file, output_file):
    df = load_data(input_file)
    df = create_normalized_key(df)
    df = generate_cleaned_text(df)
    df.to_csv("files/test.csv", index=False)
    save_data(df, output_file)


def save_data(df, output_file):
    df = df.copy()
    df = df[["raw_text", "cleaned_text"]]
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main(input_file="files/input.txt", output_file="files/output.txt")
